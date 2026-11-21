"""Stage faces / cluster / build-gallery / reid: định danh người bằng khuôn mặt (InsightFace).

Luồng:
1. ``BiometricReID.extract_faces``: phát hiện mặt trên ảnh gốc, lấy embedding ArcFace 512 chiều,
   gán mỗi mặt cho box người chứa nó.
2. ``cluster_faces`` + ``write_cluster_sheets``: gom cụm mọi khuôn mặt để người dùng đặt tên cụm.
3. ``build_gallery``: chép vài mặt tiêu biểu của mỗi cụm đã đặt tên vào identity_gallery/<Tên>/.
4. ``BiometricReID.load_gallery`` + ``match_faces``: cosine với gallery; >= tau_face thì gán tên.

Model chỉ làm việc so khớp embedding; tên người do người dùng tự đặt cho cụm/gallery.
Các hàm so khớp, gán mặt-người, gom cụm là hàm thuần trên numpy, test được không cần InsightFace.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import yaml
from PIL import Image

from .schemas import UNKNOWN_IDENTITY, BBox, Detection, FaceFile, FaceRecord, IdentityFile, IdentityMatch
from .utils import IMAGE_EXTENSIONS, Box, box_area, contains_point, get_logger, read_json, write_json

NOISE_CLUSTER = -1


# --------------------------------------------------------------------------- hàm thuần
def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def assign_face_to_person(face_box: Box, persons: Sequence[Detection]) -> Optional[str]:
    """Gán mặt cho box người NHỎ NHẤT chứa tâm mặt (người đứng trước che người đứng sau)."""
    cx, cy = (face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2
    candidates = [p for p in persons if contains_point(p.box.as_tuple(), cx, cy)]
    if not candidates:
        return None
    return min(candidates, key=lambda p: box_area(p.box.as_tuple())).entity_id


def best_gallery_match(
    embedding: np.ndarray, gallery_names: Sequence[str], gallery_matrix: np.ndarray
) -> tuple[Optional[str], Optional[float]]:
    """Tên và cosine cao nhất trong gallery (mỗi tên có thể có nhiều ảnh tham chiếu)."""
    if len(gallery_names) == 0:
        return None, None
    sims = gallery_matrix @ l2_normalize(embedding)
    idx = int(np.argmax(sims))
    return gallery_names[idx], float(sims[idx])


def match_faces(
    face_file: FaceFile,
    gallery_names: Sequence[str],
    gallery_matrix: np.ndarray,
    tau_face: float,
) -> IdentityFile:
    """So mọi mặt trong một ảnh với gallery.

    Một người chỉ xuất hiện một lần trong một ảnh: nếu hai mặt cùng khớp một tên, chỉ mặt có
    cosine cao hơn giữ tên, mặt còn lại về Unknown.
    """
    matches: list[IdentityMatch] = []
    for face in face_file.faces:
        name, sim = best_gallery_match(np.asarray(face.embedding, dtype=np.float32), gallery_names, gallery_matrix)
        identity = name if (name is not None and sim is not None and sim >= tau_face) else UNKNOWN_IDENTITY
        matches.append(
            IdentityMatch(
                entity_id=face.entity_id,
                face_id=face.face_id,
                identity=identity,
                similarity=None if sim is None else round(sim, 4),
                best_candidate=name,
            )
        )
    best_per_name: dict[str, IdentityMatch] = {}
    for m in matches:
        if m.identity == UNKNOWN_IDENTITY:
            continue
        current = best_per_name.get(m.identity)
        if current is None or (m.similarity or 0.0) > (current.similarity or 0.0):
            best_per_name[m.identity] = m
    for m in matches:
        if m.identity != UNKNOWN_IDENTITY and best_per_name[m.identity] is not m:
            m.identity = UNKNOWN_IDENTITY
    return IdentityFile(image_id=face_file.image_id, tau_face=tau_face, matches=matches)


def cluster_embeddings(embeddings: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN theo khoảng cách cosine. Trả nhãn cụm cho từng dòng; -1 là nhiễu.

    Nhãn được đánh lại để cụm 0 là cụm đông nhất.
    """
    if len(embeddings) == 0:
        return np.zeros((0,), dtype=int)
    from sklearn.cluster import DBSCAN

    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(l2_normalize(embeddings))
    ids, counts = np.unique(labels[labels != NOISE_CLUSTER], return_counts=True)
    remap = {int(old): new for new, old in enumerate(ids[np.argsort(-counts, kind="stable")])}
    return np.array([remap.get(int(label), NOISE_CLUSTER) for label in labels], dtype=int)


def rank_cluster_members(embeddings: np.ndarray, det_scores: np.ndarray) -> np.ndarray:
    """Xếp các mặt trong một cụm: gần tâm cụm và phát hiện rõ thì đứng trước."""
    normed = l2_normalize(embeddings)
    centroid = l2_normalize(normed.mean(axis=0))
    return np.argsort(-(normed @ centroid) * det_scores, kind="stable")


# --------------------------------------------------------------------------- đọc dữ liệu mặt
def load_all_faces(faces_dir: Path, min_det_score: float = 0.0, min_face_px: float = 0.0) -> list[tuple[str, FaceRecord]]:
    """Đọc mọi file output_jsons/faces/<image_id>.json, lọc theo điểm phát hiện và kích thước."""
    out: list[tuple[str, FaceRecord]] = []
    for path in sorted(faces_dir.glob("*.json")):
        face_file = FaceFile.model_validate(read_json(path))
        for face in face_file.faces:
            if face.det_score < min_det_score:
                continue
            if min(face.box.width, face.box.height) < min_face_px:
                continue
            out.append((face_file.image_id, face))
    return out


def crop_face(image: Image.Image, box: BBox, margin: float) -> Image.Image:
    """Cắt mặt kèm lề (tỉ lệ theo cạnh box) để còn tóc/cằm, giúp phát hiện lại được mặt."""
    dx, dy = box.width * margin, box.height * margin
    left, top = max(0, int(box.x1 - dx)), max(0, int(box.y1 - dy))
    right, bottom = min(image.width, int(box.x2 + dx)), min(image.height, int(box.y2 + dy))
    return image.crop((left, top, right, bottom))


# --------------------------------------------------------------------------- stage cluster
def cluster_faces(cfg: dict[str, Any], faces_dir: Path, images_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Gom cụm toàn bộ khuôn mặt, xuất contact sheet và file đặt tên cụm.

    Tạo trong ``out_dir``:
    - ``clusters.json``: mỗi cụm kèm danh sách (image_id, face_id) đã xếp hạng.
    - ``cluster_XX.jpg``: ảnh ghép các mặt tiêu biểu của cụm.
    - ``cluster_names.yaml``: người dùng điền tên; tên đã điền ở lần chạy trước được giữ lại
      nếu cụm cùng số vẫn tồn tại (đổi tham số gom cụm thì nên kiểm tra lại).
    """
    log = get_logger()
    ccfg = cfg["reid"]["cluster"]
    faces = load_all_faces(faces_dir, float(ccfg["min_det_score"]), float(ccfg["min_face_px"]))
    if not faces:
        raise RuntimeError(f"Không có khuôn mặt nào trong {faces_dir}. Chạy stage `faces` trước.")

    embeddings = np.asarray([f.embedding for _, f in faces], dtype=np.float32)
    det_scores = np.asarray([f.det_score for _, f in faces], dtype=np.float32)
    labels = cluster_embeddings(embeddings, float(ccfg["eps"]), int(ccfg["min_samples"]))
    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    log.info("%d mặt -> %d cụm, %d mặt nhiễu", len(faces), n_clusters, int((labels == NOISE_CLUSTER).sum()))

    out_dir.mkdir(parents=True, exist_ok=True)
    max_clusters, per_sheet = int(ccfg["max_clusters"]), int(ccfg["faces_per_sheet"])
    clusters: dict[str, Any] = {}
    for cluster in range(min(n_clusters, max_clusters)):
        member_idx = np.flatnonzero(labels == cluster)
        order = member_idx[rank_cluster_members(embeddings[member_idx], det_scores[member_idx])]
        key = f"cluster_{cluster:02d}"
        clusters[key] = {
            "size": int(len(member_idx)),
            "members": [{"image_id": faces[i][0], "face_id": faces[i][1].face_id} for i in order],
        }
        _write_contact_sheet([faces[i] for i in order[:per_sheet]], images_dir, out_dir / f"{key}.jpg")
    write_json(out_dir / "clusters.json", clusters)

    names_path = out_dir / "cluster_names.yaml"
    previous: dict[str, str] = {}
    if names_path.exists():
        previous = {k: v or "" for k, v in (yaml.safe_load(names_path.read_text(encoding="utf-8")) or {}).items()}
    names = {key: previous.get(key, "") for key in clusters}
    header = (
        "# Điền tên cho các cụm bạn nhận ra, ví dụ: cluster_03: \"Thầy Linh\"\n"
        "# Nhiều cụm có thể cùng một tên. Để trống = bỏ qua cụm đó.\n"
        "# Xem ảnh cluster_XX.jpg cùng thư mục. Sau đó chạy: python main.py build-gallery\n"
    )
    names_path.write_text(header + yaml.safe_dump(names, allow_unicode=True, sort_keys=True), encoding="utf-8")
    return {"faces": len(faces), "clusters": n_clusters, "sheets": len(clusters), "names_file": str(names_path)}


def _write_contact_sheet(members: list[tuple[str, FaceRecord]], images_dir: Path, out_path: Path, cell: int = 128) -> None:
    """Ghép các mặt của một cụm thành một ảnh lưới để người dùng nhìn và đặt tên."""
    if not members:
        return
    cols = min(8, len(members))
    rows = (len(members) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (32, 32, 32))
    for i, (image_id, face) in enumerate(members):
        src = _find_image(images_dir, image_id)
        if src is None:
            continue
        with Image.open(src) as im:
            thumb = crop_face(im.convert("RGB"), face.box, margin=0.4).resize((cell, cell), Image.LANCZOS)
        sheet.paste(thumb, ((i % cols) * cell, (i // cols) * cell))
    sheet.save(out_path, quality=90)


def _find_image(images_dir: Path, image_id: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------- stage build-gallery
def build_gallery(cfg: dict[str, Any], clusters_dir: Path, faces_dir: Path, images_dir: Path, gallery_dir: Path) -> dict[str, int]:
    """Chép các mặt tiêu biểu của những cụm đã đặt tên vào ``identity_gallery/<Tên>/``.

    Thư mục của mỗi tên được dựng lại từ đầu; ảnh tham chiếu người dùng tự thêm nên đặt tên file
    bắt đầu bằng ``manual_`` để không bị xoá.
    """
    log = get_logger()
    names_path = clusters_dir / "cluster_names.yaml"
    if not names_path.exists():
        raise FileNotFoundError(f"Không thấy {names_path}. Chạy stage `cluster` trước.")
    names: dict[str, str] = yaml.safe_load(names_path.read_text(encoding="utf-8")) or {}
    clusters: dict[str, Any] = read_json(clusters_dir / "clusters.json")
    refs = int(cfg["reid"]["gallery_refs_per_identity"])

    # Nhiều cụm cùng tên: lấy xen kẽ từ các cụm để gallery phủ nhiều góc mặt.
    by_name: dict[str, list[list[dict[str, str]]]] = {}
    for key, name in names.items():
        name = (name or "").strip()
        if name and key in clusters:
            by_name.setdefault(name, []).append(clusters[key]["members"])

    written: dict[str, int] = {}
    for name, member_lists in by_name.items():
        picked: list[dict[str, str]] = []
        rank = 0
        while len(picked) < refs and any(rank < len(m) for m in member_lists):
            for members in member_lists:
                if rank < len(members) and len(picked) < refs:
                    picked.append(members[rank])
            rank += 1

        person_dir = gallery_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)
        for old in person_dir.iterdir():
            if old.is_file() and not old.name.startswith("manual_"):
                old.unlink()
        count = 0
        for member in picked:
            src = _find_image(images_dir, member["image_id"])
            face_path = faces_dir / f"{member['image_id']}.json"
            if src is None or not face_path.exists():
                continue
            face_file = FaceFile.model_validate(read_json(face_path))
            face = next((f for f in face_file.faces if f.face_id == member["face_id"]), None)
            if face is None:
                continue
            with Image.open(src) as im:
                crop_face(im.convert("RGB"), face.box, margin=1.0).save(
                    person_dir / f"{member['image_id']}_{member['face_id']}.jpg", quality=95
                )
            count += 1
        written[name] = count
        log.info("Gallery %s: %d ảnh tham chiếu", name, count)
    if not written:
        log.warning("Chưa có cụm nào được đặt tên trong %s", names_path)
    return written


# --------------------------------------------------------------------------- model
class BiometricReID:
    """Bọc InsightFace ``FaceAnalysis``: phát hiện mặt + embedding ArcFace."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from insightface.app import FaceAnalysis

        self.cfg = cfg
        self.log = get_logger()
        rcfg = cfg["reid"]
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        kwargs: dict[str, Any] = {"name": rcfg["model_name"], "providers": providers}
        if cfg["paths"].get("model_cache"):
            kwargs["root"] = cfg["paths"]["model_cache"]
        self.app = FaceAnalysis(**kwargs)
        # ctx_id=0: GPU đầu tiên nếu onnxruntime-gpu dùng được, nếu không tự rơi về CPU.
        self.app.prepare(ctx_id=0, det_size=tuple(int(v) for v in rcfg["det_size"]))
        self.min_det_score = float(rcfg["min_det_score"])

    def _detect(self, image_path: Path) -> list[Any]:
        import cv2

        # imdecode + fromfile đọc được đường dẫn có ký tự không phải ASCII trên Windows.
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")
        return list(self.app.get(image))

    def extract_faces(self, image_path: Path, image_id: str, persons: Sequence[Detection]) -> FaceFile:
        """Mọi khuôn mặt trong một ảnh, đã gán cho box người (nếu có)."""
        records: list[FaceRecord] = []
        faces = sorted(self._detect(image_path), key=lambda f: float(f.bbox[0]))
        for index, face in enumerate(faces, start=1):
            det_score = float(face.det_score)
            if det_score < self.min_det_score:
                continue
            box = BBox.from_seq(face.bbox)
            records.append(
                FaceRecord(
                    face_id=f"f{index:02d}",
                    box=box,
                    det_score=round(det_score, 4),
                    entity_id=assign_face_to_person(box.as_tuple(), persons),
                    embedding=[round(float(v), 5) for v in face.normed_embedding],
                )
            )
        return FaceFile(image_id=image_id, faces=records)

    def load_gallery(self, gallery_dir: Path) -> tuple[list[str], np.ndarray]:
        """Embedding tham chiếu từ ``identity_gallery/<Tên>/*``; mỗi ảnh lấy mặt lớn nhất.

        Trả (tên cho từng dòng, ma trận N x 512 đã chuẩn hoá). Gallery rỗng trả ([], ma trận 0 dòng).
        """
        names: list[str] = []
        vectors: list[np.ndarray] = []
        if gallery_dir.is_dir():
            for person_dir in sorted(p for p in gallery_dir.iterdir() if p.is_dir()):
                found = 0
                for image_path in sorted(person_dir.iterdir()):
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    faces = self._detect(image_path)
                    if not faces:
                        self.log.warning("Gallery: không thấy mặt trong %s", image_path)
                        continue
                    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    names.append(person_dir.name)
                    vectors.append(np.asarray(largest.normed_embedding, dtype=np.float32))
                    found += 1
                self.log.info("Gallery %s: %d ảnh tham chiếu dùng được", person_dir.name, found)
        matrix = l2_normalize(np.stack(vectors)) if vectors else np.zeros((0, 512), dtype=np.float32)
        return names, matrix


def summarize_similarities(identity_files: Sequence[IdentityFile], tau_face: float) -> str:
    """Phân bố cosine cao nhất của mọi mặt, để người dùng xem ngưỡng tau_face có hợp lý không."""
    sims = np.asarray([m.similarity for f in identity_files for m in f.matches if m.similarity is not None])
    if sims.size == 0:
        return "Không có điểm tương đồng nào (gallery rỗng hoặc không có khuôn mặt)."
    edges = [-1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 1.01]
    counts, _ = np.histogram(sims, bins=edges)
    lines = [f"Phân bố cosine cao nhất của {sims.size} mặt (tau_face = {tau_face}):"]
    for low, high, count in zip(edges[:-1], edges[1:], counts):
        lines.append(f"  [{max(low, 0):.2f}, {min(high, 1):.2f}): {int(count)}")
    named = sum(1 for f in identity_files for m in f.matches if m.identity != UNKNOWN_IDENTITY)
    lines.append(f"Số mặt được gán tên: {named}")
    return "\n".join(lines)
