from PIL import Image

images = [
    Image.open(x)
    for x in [
        "../figs/grid_search_perceptron_ablation.png",
        "../figs/grid_search_multinomial_nb_ablation.png",
        "../figs/grid_search_decision_tree_ablation.png",
    ]
]
widths, heights = zip(*(i.size for i in images))
new_img = Image.new("RGB", (sum(widths), max(heights)))

x_offset = 0
for img in images:
    new_img.paste(img, (x_offset, 0))
    x_offset += img.size[0]


new_img.save("../figs/grid_search_results.png")
