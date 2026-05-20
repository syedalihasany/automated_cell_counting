
import os
import glob
import traceback
import numpy as np
import pandas as pd
import tifffile
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tkinter.scrolledtext import ScrolledText

from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, feature


def load_two_channel_tiff(path):
    img = tifffile.imread(path)

    if img.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF with 2 or 3 channels, got shape {img.shape}")

    # Case 1: channels-first, 2-channel
    if img.shape[0] == 2:
        red = img[0].astype(np.float32)
        green = img[1].astype(np.float32)

    # Case 2: channels-last, 2-channel
    elif img.shape[-1] == 2:
        red = img[..., 0].astype(np.float32)
        green = img[..., 1].astype(np.float32)

    # Case 3: channels-first, RGB TIFF
    elif img.shape[0] == 3:
        red = img[0].astype(np.float32)
        green = img[1].astype(np.float32)

    # Case 4: channels-last, RGB TIFF
    elif img.shape[-1] == 3:
        red = img[..., 0].astype(np.float32)
        green = img[..., 1].astype(np.float32)

    else:
        raise ValueError(f"Could not identify usable channels in TIFF shape {img.shape}")

    return red, green

def normalize_channel(ch):
    ch = ch.astype(np.float32)
    ch = ch - ch.min()
    if ch.max() > 0:
        ch = ch / ch.max()
    return ch


def background_correct(ch, sigma=12):
    bg = filters.gaussian(ch, sigma=sigma)
    corr = ch - bg
    corr = np.clip(corr, 0, None)
    if corr.max() > 0:
        corr = corr / corr.max()
    return corr


def make_mask(
    ch_corr,
    threshold_floor=0.18,
    method="otsu",
    opening_radius=1,
    closing_radius=1,
    min_size=12
):
    if method == "otsu":
        vals = ch_corr[ch_corr > 0]
        if len(vals) > 0:
            thr = filters.threshold_otsu(vals)
        else:
            thr = 1.0
        thr = max(thr, threshold_floor)
    else:
        thr = threshold_floor

    mask = ch_corr > thr

    if opening_radius > 0:
        mask = morphology.opening(mask, morphology.disk(opening_radius))
    if closing_radius > 0:
        mask = morphology.closing(mask, morphology.disk(closing_radius))

    mask = morphology.remove_small_objects(mask, min_size=min_size)
    return mask, thr


def count_objects_watershed(mask, min_distance=4, min_area=12, max_area=400):
    distance = ndi.distance_transform_edt(mask)
    coords = feature.peak_local_max(distance, min_distance=min_distance, labels=mask)

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if markers.max() == 0:
        labels = np.zeros_like(mask, dtype=np.int32)
    else:
        labels = segmentation.watershed(-distance, markers, mask=mask)

    count = 0
    for prop in measure.regionprops(labels):
        if min_area <= prop.area <= max_area:
            count += 1
    return count


def count_objects_connected(mask, min_area=12, max_area=400, connectivity=2):
    labeled = measure.label(mask, connectivity=connectivity)
    count = 0
    for prop in measure.regionprops(labeled):
        if min_area <= prop.area <= max_area:
            count += 1
    return count


def count_blobs_log(
    image_for_blobs,
    mask=None,
    min_sigma=1.5,
    max_sigma=5.0,
    num_sigma=10,
    threshold=0.05,
    overlap=0.5
):
    blobs = feature.blob_log(
        image_for_blobs,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap
    )

    count = 0
    h, w = image_for_blobs.shape
    for y, x, _ in blobs:
        yi = int(round(y))
        xi = int(round(x))
        if yi < 0 or yi >= h or xi < 0 or xi >= w:
            continue
        if mask is not None and not mask[yi, xi]:
            continue
        count += 1
    return count



def load_single_channel_image(path):
    """
    Load a single-channel fluorescence image from TIFF/PNG/JPG.
    If RGB, convert to grayscale by taking the maximum across channels.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        img = tifffile.imread(path)
    else:
        try:
            from imageio import v2 as imageio
            img = imageio.imread(path)
        except Exception as e:
            raise ValueError(f"Unsupported or unreadable image: {path}") from e

    img = np.asarray(img)

    if img.ndim == 2:
        ch = img.astype(np.float32)
    elif img.ndim == 3:
        if img.shape[0] == 1:
            ch = img[0].astype(np.float32)
        elif img.shape[-1] == 1:
            ch = img[..., 0].astype(np.float32)
        else:
            # For RGB-like data, use maximum intensity projection.
            if img.shape[0] <= 4 and img.shape[-1] > 4:
                ch = img.max(axis=0).astype(np.float32)
            else:
                ch = img.max(axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape for single-channel mode: {img.shape}")

    return ch

DEFAULT_PARAMS = {
    "bg_sigma": 12,
    "red_floor": 0.18,
    "green_floor": 0.18,
    "yellow_floor": 0.15,
    "min_size_mask": 12,
    "opening_radius": 1,
    "closing_radius": 1,
    "min_distance": 4,
    "min_area": 12,
    "max_area": 400,
    "connectivity": 2,
    "min_sigma": 1.5,
    "max_sigma": 5.0,
    "num_sigma": 10,
    "blob_threshold": 0.05,
    "blob_overlap": 0.5,
}


def analyze_tiff(tiff_path, method_name, params=None):
    if params is None:
        params = DEFAULT_PARAMS

    red_raw, green_raw = load_two_channel_tiff(tiff_path)

    red_raw_n = normalize_channel(red_raw)
    green_raw_n = normalize_channel(green_raw)

    red_corr = background_correct(red_raw_n, sigma=params["bg_sigma"])
    green_corr = background_correct(green_raw_n, sigma=params["bg_sigma"])

    red_mask, red_thr = make_mask(
        red_corr,
        threshold_floor=params["red_floor"],
        method="otsu",
        opening_radius=params["opening_radius"],
        closing_radius=params["closing_radius"],
        min_size=params["min_size_mask"]
    )

    green_mask, green_thr = make_mask(
        green_corr,
        threshold_floor=params["green_floor"],
        method="otsu",
        opening_radius=params["opening_radius"],
        closing_radius=params["closing_radius"],
        min_size=params["min_size_mask"]
    )

    yellow_signal = np.minimum(red_corr, green_corr)
    yellow_mask, yellow_thr = make_mask(
        yellow_signal,
        threshold_floor=params["yellow_floor"],
        method="otsu",
        opening_radius=params["opening_radius"],
        closing_radius=params["closing_radius"],
        min_size=params["min_size_mask"]
    )

    if method_name == "Watershed":
        red_count = count_objects_watershed(
            red_mask,
            min_distance=params["min_distance"],
            min_area=params["min_area"],
            max_area=params["max_area"]
        )
        green_count = count_objects_watershed(
            green_mask,
            min_distance=params["min_distance"],
            min_area=params["min_area"],
            max_area=params["max_area"]
        )
        yellow_count = count_objects_watershed(
            yellow_mask,
            min_distance=params["min_distance"],
            min_area=params["min_area"],
            max_area=params["max_area"]
        )
    elif method_name == "Simply Connected":
        red_count = count_objects_connected(
            red_mask,
            min_area=params["min_area"],
            max_area=params["max_area"],
            connectivity=params["connectivity"]
        )
        green_count = count_objects_connected(
            green_mask,
            min_area=params["min_area"],
            max_area=params["max_area"],
            connectivity=params["connectivity"]
        )
        yellow_count = count_objects_connected(
            yellow_mask,
            min_area=params["min_area"],
            max_area=params["max_area"],
            connectivity=params["connectivity"]
        )
    elif method_name == "LoG / Blob Detection":
        red_count = count_blobs_log(
            red_corr,
            mask=red_mask,
            min_sigma=params["min_sigma"],
            max_sigma=params["max_sigma"],
            num_sigma=params["num_sigma"],
            threshold=params["blob_threshold"],
            overlap=params["blob_overlap"]
        )
        green_count = count_blobs_log(
            green_corr,
            mask=green_mask,
            min_sigma=params["min_sigma"],
            max_sigma=params["max_sigma"],
            num_sigma=params["num_sigma"],
            threshold=params["blob_threshold"],
            overlap=params["blob_overlap"]
        )
        yellow_count = count_blobs_log(
            yellow_signal,
            mask=yellow_mask,
            min_sigma=params["min_sigma"],
            max_sigma=params["max_sigma"],
            num_sigma=params["num_sigma"],
            threshold=params["blob_threshold"],
            overlap=params["blob_overlap"]
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    union_area = int((red_mask | green_mask).sum())
    yellow_area = int(yellow_mask.sum())
    yellow_over_union_pct = 100 * yellow_area / union_area if union_area > 0 else 0.0

    return {
        "file": os.path.basename(tiff_path),
        "method": method_name,
        "Red count": red_count,
        "Green count": green_count,
        "Yellow/common count": yellow_count,
        "Yellow over Union (%)": yellow_over_union_pct,
        "red_threshold_used": red_thr,
        "green_threshold_used": green_thr,
        "yellow_threshold_used": yellow_thr,
    }



def analyze_single_channel_image(image_path, method_name, params=None):
    """
    Analyze an image that contains only one fluorescence channel.
    Returns only the count of structures in that channel.
    """
    if params is None:
        params = DEFAULT_PARAMS

    raw = load_single_channel_image(image_path)
    raw_n = normalize_channel(raw)
    corr = background_correct(raw_n, sigma=params["bg_sigma"])

    single_mask, thr = make_mask(
        corr,
        threshold_floor=params["red_floor"],
        method="otsu",
        opening_radius=params["opening_radius"],
        closing_radius=params["closing_radius"],
        min_size=params["min_size_mask"]
    )

    if method_name == "Watershed":
        count = count_objects_watershed(
            single_mask,
            min_distance=params["min_distance"],
            min_area=params["min_area"],
            max_area=params["max_area"]
        )
    elif method_name == "Simply Connected":
        count = count_objects_connected(
            single_mask,
            min_area=params["min_area"],
            max_area=params["max_area"],
            connectivity=params["connectivity"]
        )
    elif method_name == "LoG / Blob Detection":
        count = count_blobs_log(
            corr,
            mask=single_mask,
            min_sigma=params["min_sigma"],
            max_sigma=params["max_sigma"],
            num_sigma=params["num_sigma"],
            threshold=params["blob_threshold"],
            overlap=params["blob_overlap"]
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    return {
        "file": os.path.basename(image_path),
        "mode": "Single fluorescence",
        "method": method_name,
        "Structure count": count,
        "threshold_used": thr,
    }

class CellCountingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TIFF Cell/Signal Counter")
        self.root.geometry("900x650")
        self.selected_files = []
        self.last_df = None

        self.mode_var = tk.StringVar(value="Dual-channel (Red/Green/Yellow)")
        self.method_var = tk.StringVar(value="Watershed")
        self.status_var = tk.StringVar(value="Select image files or a folder, choose a mode and method, then run.")

        top = ttk.Frame(root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="Mode:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.mode_combo = ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=["Dual-channel (Red/Green/Yellow)", "Single fluorescence only"],
            state="readonly",
            width=28
        )
        self.mode_combo.grid(row=0, column=1, sticky="w", padx=(0, 12))

        ttk.Label(top, text="Counting method:").grid(row=0, column=2, sticky="w", padx=(0, 8))
        self.method_combo = ttk.Combobox(
            top,
            textvariable=self.method_var,
            values=["Watershed", "Simply Connected", "LoG / Blob Detection"],
            state="readonly",
            width=24
        )
        self.method_combo.grid(row=0, column=3, sticky="w")

        ttk.Button(top, text="Select image files", command=self.select_files).grid(row=0, column=4, padx=8)
        ttk.Button(top, text="Select folder", command=self.select_folder).grid(row=0, column=5, padx=8)
        ttk.Button(top, text="Clear selection", command=self.clear_selection).grid(row=0, column=6, padx=8)

        middle = ttk.LabelFrame(root, text="Selected files", padding=12)
        middle.pack(fill="both", expand=False, padx=12, pady=8)

        self.file_list = ScrolledText(middle, height=10, wrap="word")
        self.file_list.pack(fill="both", expand=True)

        controls = ttk.Frame(root, padding=12)
        controls.pack(fill="x")

        ttk.Button(controls, text="Run analysis", command=self.run_analysis).pack(side="left")
        ttk.Button(controls, text="Save last results as CSV", command=self.save_csv).pack(side="left", padx=10)
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=12)

        bottom = ttk.LabelFrame(root, text="Results", padding=12)
        bottom.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.results_box = ScrolledText(bottom, wrap="word")
        self.results_box.pack(fill="both", expand=True)

    def log(self, text=""):
        self.results_box.insert("end", text + "\n")
        self.results_box.see("end")
        self.root.update_idletasks()

    def refresh_file_list(self):
        self.file_list.delete("1.0", "end")
        if not self.selected_files:
            self.file_list.insert("end", "No files selected.\n")
            return
        for f in self.selected_files:
            self.file_list.insert("end", f + "\n")

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select image files",
            filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if files:
            self.selected_files = list(files)
            self.refresh_file_list()
            self.status_var.set(f"{len(self.selected_files)} file(s) selected.")

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing image files")
        if not folder:
            return
        image_files = (
            sorted(glob.glob(os.path.join(folder, "*.tif"))) +
            sorted(glob.glob(os.path.join(folder, "*.tiff"))) +
            sorted(glob.glob(os.path.join(folder, "*.png"))) +
            sorted(glob.glob(os.path.join(folder, "*.jpg"))) +
            sorted(glob.glob(os.path.join(folder, "*.jpeg")))
        )
        self.selected_files = image_files
        self.refresh_file_list()
        self.status_var.set(f"{len(self.selected_files)} image file(s) found in folder.")

    def clear_selection(self):
        self.selected_files = []
        self.refresh_file_list()
        self.status_var.set("Selection cleared.")

    def run_analysis(self):
        if not self.selected_files:
            messagebox.showwarning("No files selected", "Please select one or more TIFF files first.")
            return

        method_name = self.method_var.get()
        self.results_box.delete("1.0", "end")
        self.log(f"Method: {method_name}")
        self.log("Using the same default parameters as before.\n")

        rows = []
        for idx, path in enumerate(self.selected_files, start=1):
            try:
                result = analyze_tiff(path, method_name, DEFAULT_PARAMS)
                rows.append(result)
                self.log(f"[{idx}/{len(self.selected_files)}] {result['file']}")
                self.log(f"Red count: {result['Red count']}")
                self.log(f"Green count: {result['Green count']}")
                self.log(f"Yellow/common count: {result['Yellow/common count']}")
                self.log(f"Yellow over Union: {result['Yellow over Union (%)']:.2f}%")
                self.log("-" * 50)
            except Exception as e:
                self.log(f"[{idx}/{len(self.selected_files)}] ERROR on {os.path.basename(path)}")
                self.log(str(e))
                self.log(traceback.format_exc())
                self.log("-" * 50)

        if rows:
            self.last_df = pd.DataFrame(rows, columns=[
                "file",
                "method",
                "Red count",
                "Green count",
                "Yellow/common count",
                "Yellow over Union (%)",
                "red_threshold_used",
                "green_threshold_used",
                "yellow_threshold_used",
            ])
            self.status_var.set(f"Done. Processed {len(rows)} file(s).")
        else:
            self.last_df = None
            self.status_var.set("No successful results.")

    def save_csv(self):
        if self.last_df is None or self.last_df.empty:
            messagebox.showinfo("No results", "Run an analysis first.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save results CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not save_path:
            return

        self.last_df.to_csv(save_path, index=False)
        self.status_var.set(f"Saved CSV to: {save_path}")
        messagebox.showinfo("Saved", f"Results saved to:\n{save_path}")


def main():
    root = tk.Tk()
    CellCountingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
