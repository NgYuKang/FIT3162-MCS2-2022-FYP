from flask import Flask, render_template, request
from PIL import Image
import base64
import io
import os

from torchvision import transforms
import torch
from torch.autograd import Variable
import torch.nn.functional as F

# Init Variables
SECRET_KEY = os.urandom(32)
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = 'uploads/'
app.config['SECRET_KEY'] = SECRET_KEY

# Allowed extensions
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")

# Transformation
transform = transforms.Compose([
    # resize
    transforms.Resize(256),
    # center_crop
    transforms.CenterCrop(224),
    transforms.GaussianBlur(3, 1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4704, 0.4565, 0.4425], std=[0.3045, 0.2898, 0.2999])
])

# Unnormalize transform
transform_unnormalize = transforms.Compose([
    transforms.Normalize(mean=[-0.4704 / 0.3045, -0.4565 / 0.2898, -0.4425 / 0.2999],
                         std=[1.0 / 0.3045, 1.0 / 0.2898, 1.0 / 0.2999])
])
# transform to convert to Pillow Image
t_pil = transforms.ToPILImage()

# Load and set model to evaluation mode
model = torch.jit.load('resnet50_sports_non_sports_multilabel.pt')
model = model.cpu()
model.eval()

# Label map
labels_map = {
    0: "Non Sports",
    1: "Sports",
}

sub_label = ['Cello',
             'Dab',
             'Dog chasing ball',
             'Driving',
             'Guitar',
             'Harp',
             'Holding ball',
             'Holding something',
             'Jumping',
             'Leisure sea activity',
             'Pedestrian',
             'Plane flying',
             'Queue',
             'Picture with sky in background',
             'Reading',
             'Riding motorcycle',
             'Ship at sea',
             'Sleeping',
             'Standing',
             'Standing beside bicycle',
             'Using computer',
             'Violin',
             'Waving',
             'Badminton',
             'Baseball',
             'Basketball',
             'Boxing',
             'Cycling',
             'Fencing',
             'Football',
             'Golf',
             'Race walking',
             'Sky diving',
             'Squash',
             'Swimming',
             'Table tennis',
             'Tennis',
             'Track and field',
             'Volleyball',
             'Weightlifting']


def predict_image(image):
    """
    Takes in an image and uses the model to predict it
    :param image: Pillow Image
    :return: Main and sub label along with the probability
    """
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    output = model(image_tensor)

    main_label_output = F.softmax(output['main_label'], dim=1)
    sub_label_output = F.softmax(output['sub_label'], dim=1)

    prediction_score, pred_label_idx = torch.topk(main_label_output, 1)
    pred_label_main = pred_label_idx.item()

    prediction_score_sub, pred_label_idx_sub = torch.topk(sub_label_output, 1)
    pred_label_sub = pred_label_idx_sub.item()

    prob_main = round(prediction_score.squeeze().item() * 100, 2)
    prob_sub = round(prediction_score_sub.squeeze().item() * 100, 2)

    return [pred_label_main, prob_main], [pred_label_sub, prob_sub]


@app.route("/", methods=['GET'])
def homepage():
    """
    Return and render the main page for GET request
    """
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_img():
    """
    Function that is invoked when POST request received on page
    """
    data = request.files['image']  # Take the image uploaded
    filetype = data.filename.split('.')[-1]  # Get the filetype
    error = ""

    # Try catch to catch image opening error
    try:
        # Check if there is any data at all, and if it is valid
        if data.filename != '' and filetype.lower() in IMG_EXTENSIONS:
            img = Image.open(data).convert('RGB')  # Convert to RGB, remove the A component
            main_res, sub_res = predict_image(img)  # Predict

            data = io.BytesIO()  # Reconvert the image into bytes or something so we can send it back to the page

            # Save and encode to put back into the website
            img.save(data, "PNG")
            encoded_img_data = base64.b64encode(data.getvalue())

            # If sports label
            if main_res[0] == 1:
                # If predicted non-sports sub-label
                if sub_res[0] <= 17:
                    sub_res[0] = "Failed to predict the image's sports sub-label as sports."
                    sub_res[1] = None
                else:
                    # Output sports sub label
                    sub_res[0] = sub_label[sub_res[0]]
            # Non sports, output no sub label
            else:
                sub_res[0] = None

            # Get main label
            main_res[0] = labels_map[main_res[0]]
            # Return template with all the data
            return render_template("index.html", img_data=encoded_img_data.decode('utf-8'), result=main_res,
                                   sub_result=sub_res)
    except OSError:
        # File opening error
        error = "Unexpected error has occurred. It may be due to the image being corrupted."

    # If invalid extension
    if filetype != '' and filetype.lower() not in IMG_EXTENSIONS:
        error = "Invalid file type was uploaded!"
    # If no file was uploaded
    elif data.filename == '':
        error = "No image was uploaded!"

    # Render template with error
    return render_template("index.html", error=error)


if __name__ == '__main__':
    app.run()
