import os
import rasterio
from rasterio.windows import Window


def compute_starts(dim_size, tile_size):
    """
    Compute start indices so that:
    - full coverage is achieved
    - last tile is exactly tile_size and overlaps if needed
    """
    starts = list(range(0, dim_size - tile_size + 1, tile_size))

    if not starts:
        return [0]

    last_start = starts[-1]
    if last_start + tile_size < dim_size:
        starts.append(dim_size - tile_size)

    return starts


def tile_tif(
    input_path,
    output_dir,
    tile_size=512,
    ext="tif"
):
    os.makedirs(output_dir, exist_ok=True)

    input_filename = os.path.splitext(os.path.basename(input_path))[0]

    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        profile = src.profile

        x_starts = compute_starts(width, tile_size)
        y_starts = compute_starts(height, tile_size)

        tile_count = 0

        for y in y_starts:
            for x in x_starts:
                window = Window(x, y, tile_size, tile_size)
                data = src.read(window=window)

                # Update transform for this tile
                tile_transform = rasterio.windows.transform(window, transform)

                tile_profile = profile.copy()
                tile_profile.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": tile_transform
                })

                out_name = f"{input_filename}_{y}_{x}.{ext}"
                out_path = os.path.join(output_dir, out_name)

                with rasterio.open(out_path, "w", **tile_profile) as dst:
                    dst.write(data)

                tile_count += 1

        print(f"Generated {tile_count} tiles.")


if __name__ == "__main__":
    input_path = "C:/Users/admin/Desktop/inference_data/Inference_data/74865_1014763_N-34-50-C-c-4-4.tif"
    output_dir = "C:/Users/admin/Desktop/inference_data/Inference_data/mck26"
    tile_size = 625

    tile_tif(
        input_path=input_path,
        output_dir=output_dir,
        tile_size=tile_size,
    )
