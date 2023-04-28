import imageio
from tqdm import tqdm

from .multidimensional import multidimensional


def gif_multidimensional(
    data,
    proposed_info={},
    labels=["X values", "Y values", "Histogram values"],
    filename="testfile",
    folder="test",
    limits=None,
    DPI=150,
    figsize=(12, 8),
):
    r"""
    Plot many multidimensional figures in their own `{folder}/{filename}/` directory. Each
    file will represent one ensemble of plans, and this will be stitched together to create
    a gif.

    Args:
        data (dict): Dictionary with keys of `xs`, `ys`, `hists`, each one a list of length
            number of frames/ensembles, where each element is a list of all values in the ensemble.
        proposed_info (dict, optional): Dictionary with keys of `colors`, `names`,
            `x`, `y`, `hist`; the \(i\)th color in `color` corresponds to the
            \(i\)th name in `names`, which corresponds to the \(i\)th value in
            `x`, `y`, and `hist`.
        filename (str): Name for the final gif, and for the folder the gif's frames are stored in
        folder (str): Folder containing all the frames/gifs.
        figsize (tuple, optional): Figure size.

    Returns: None.
    """
    # Get the x- and y-values.
    xs = data["xs"]
    ys = data["ys"]

    # TODO: Check what's actually happening here.
    hists = data["hists"]
    assert len(xs) == len(ys) == len(hists)

    lower_bound = 1
    upper_bound = 1
    x_limits = (lower_bound * min_of_min(xs), upper_bound * max_of_max(xs))
    y_limits = (lower_bound * min_of_min(ys), upper_bound * max_of_max(ys))
    hist_limits = (
        int(lower_bound * min_of_min(hists)),
        int(upper_bound * max_of_max(hists)),
    )
    limits = [x_limits, y_limits, hist_limits] if not limits else limits

    save = {
        "folder": folder,
        "filename": filename,
        "DPI": DPI,
    }

    print("Generating PNGs...")

    for i in tqdm(range(len(xs))):
        save["frame"] = i
        multidimensional(
            xs[i],
            ys[i],
            hists[i],
            labels=labels,
            limits=limits,
            proposed_info=proposed_info,
            figsize=figsize,
            save=save,
        )

    print("Generating gif...")
    images = []
    path = f"{save['folder']}/{save['filename']}/{save['filename']}_"
    output = f"{save['folder']}/{save['filename']}.gif"
    for i in tqdm(range(len(xs))):
        images.append(imageio.imread(f"{path}{i}.png"))
    imageio.mimsave(output, images)


def min_of_min(big_list):
    return min([min(sub_list) for sub_list in big_list])


def max_of_max(big_list):
    return max([max(sub_list) for sub_list in big_list])
