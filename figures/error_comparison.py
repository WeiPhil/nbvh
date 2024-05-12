# %%
import figuregen
from figuregen.util import image
from figuregen import util
import simpleimageio as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
%matplotlib inline

plt.rcParams.update({
    "text.usetex": True,     
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    "pgf.texsystem": "pdflatex",
    "font.family": "sans-serif",
    "pgf.preamble": "\n".join([
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage" + "{libertine}"
        ])
})

# %%

from utils import compute_error

experiment_name = "normal_experiments"
input_dir = "../" + experiment_name + "/"

method_names = ["reference",
                "linear_activated_normals_3_points_visibility_masked",
                "sigmoid_activated_normals_3_points"]

reference_method_names = ["reference",
                         "reference",
                         "reference"]

assert(len(reference_method_names) == len(method_names))
display_method_names = [method_name.replace('hash','').replace('_',' ').capitalize() for method_name in method_names]
display_method_names[0] = "Reference"
display_method_names[1] = "Linear Activation"
display_method_names[2] = "Sigmoid Activation"
# display_method_names[2] = "Ours (3 point segment)"
# display_method_names[3] = "Ours (5 point segment)"
# display_method_names[4] = "Ours (10 point segment)"

method_losses = []
image_grid = [] 
ref_image_grid = [] 
error_image_grid = [] 

crop_box = image.Cropbox(0,568,1080,1920-2*568)

error_type = "FLIP" 
# "FLIP"

for method_name, reference_method_name in zip(method_names,reference_method_names):
    row = []
    ref_row = []
    error_row = []
 
    ref_filename = input_dir + reference_method_name + ".exr"
    impostor_filename = input_dir +  method_name + ".exr"
    print(ref_filename)
    print(impostor_filename)
    ref_image = crop_box.crop(sio.read(ref_filename))
    ref_image = image.lin_to_srgb(crop_box.crop(sio.read(ref_filename)))
    impostor_image = crop_box.crop(sio.read(impostor_filename))
    impostor_image = image.lin_to_srgb(crop_box.crop(sio.read(impostor_filename)))

    if len(impostor_image.shape) == 2:
        impostor_image = np.repeat(np.expand_dims(impostor_image, axis=-1), 3, axis=-1)

    if len(ref_image.shape) == 2:
        ref_image = np.repeat(np.expand_dims(ref_image, axis=-1), 3, axis=-1)
        
    ref_row.append(ref_image)
    row.append(impostor_image)
    error_row.append(compute_error(impostor_image,ref_image,error_type))

    image_grid.append(row)
    ref_image_grid.append(ref_row)
    error_image_grid.append(error_row)

print(len(image_grid))


# %%


num_rows = 1
num_cols = len(image_grid)
impostor_grid = figuregen.Grid(num_rows=num_rows, num_cols=num_cols)
# impostor_grid.set_title(position="top",txt_content =r"\hspace{2.1cm}\textsc{Dragon}\hspace{2.1cm}")

for row in range(num_rows):
    for col in range(num_cols):
        e = impostor_grid.get_element(row, col)  
        img = image_grid[col][row]
        e.set_image(figuregen.PNG(img))
        e.set_frame(1)

main_layout = impostor_grid.get_layout().set_padding(bottom=1)
main_layout.set_row_titles('left', field_size_mm=6.)

row_titles = ["Impostor"] 
column_titles = display_method_names

impostor_grid.set_col_titles("top",column_titles)
impostor_grid.set_row_titles("left",row_titles)
# ****************

num_rows = 1
num_cols = len(error_image_grid)
error_grid = figuregen.Grid(num_rows=num_rows, num_cols=num_cols)

baseline_error = np.mean(error_image_grid[0][0])
for row in range(num_rows):
    for col in range(num_cols):
        e = error_grid.get_element(row, col)
  
        img = error_image_grid[col][row]
 
        
        # if col == num_cols - 1:
        #     e.set_image(img)
        # else:
        e.set_image(figuregen.PNG(img))
        e.set_frame(1)
        mean_error = np.mean(error_image_grid[col][row])
        error_ratio = baseline_error / mean_error 
        e.set_caption(r"\textit{" + "{:.4f}".format(mean_error)+"}" + r"\textit{\textbf{"+ " ({:.2f}x)".format(error_ratio) +"}}")


# main_layout = main_grid.get_layout().set_padding(row=0.3,column=1)
main_layout = error_grid.get_layout()
main_layout.set_row_titles('left', field_size_mm=6.)
main_layout.set_caption(height_mm=3.4,fontsize=8,offset_mm=0.5)

row_titles = [error_type + " Error"] 
error_grid.set_row_titles("left",row_titles)

grids = [[impostor_grid],[error_grid]]

# %%

if __name__ == "__main__":
    width_cm = 16
    # figuregen.figure(grids, width_cm=width_cm, filename='parameterization_rayfoot_{}.pdf'.format(scene_name),
    figuregen.figure(grids, width_cm=width_cm, filename= experiment_name+'.pdf',
                     backend=figuregen.PdfBackend(None, [
                         "\\usepackage[utf8]{inputenc}",
                         "\\usepackage[T1]{fontenc}",
                        #  "\\usepackage{libertine}"
                         "\\usepackage{color}"
                         "\\usepackage{xparse}"
                         "\\usepackage[outline]{contour}"
                         "\\contourlength{.04em}"
                         "\\usepackage[many]{tcolorbox}"
                         "\\usepackage{mathtools}"
                         "\\usepackage{xspace}"
                         "\\newcommand{\\FLIP}{\\protect\\reflectbox{F}LIP\\xspace}"
                     ]))
    try:
        from figuregen.util import jupyter
        jupyter.convert(experiment_name+'.pdf', 1200)
    except:
        print('Warning: pdf could not be converted to png')

# %%
