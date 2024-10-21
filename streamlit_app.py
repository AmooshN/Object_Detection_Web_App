import streamlit as st
from objectDetection import *
from PIL import Image


st.title("Object Detection Web App")

def func_1(model_type):
    detector = Detector(model_type=model_type)
    image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
    
    if image_file is not None:
        # Save the uploaded image
        with open(image_file.name, mode="wb") as f:
            f.write(image_file.getbuffer())
        
        # Process the image using the detector
        detector.onImage(image_file.name)

        # Load both the original and processed images
        original_img = Image.open(image_file)
        processed_img = Image.open("result.jpg")

        # Dropdown for image selection
        option = st.selectbox(
            "Select Image to View:",
            ("Processed Image", "Original Image"),
            index=0  # Default to showing the processed image first
        )

        # Display the selected image
        if option == "Processed Image":
            st.image(processed_img, caption='Processed Image.')
        else:
            st.image(original_img, caption='Original Image.')

        # Provide download option for the result image
        with open("result.jpg", "rb") as file:
            btn = st.download_button(
                label="Download Processed Image",
                data=file,
                file_name="processed_result.jpg",
                mime="image/jpeg"
            )

def func_2(model_type):
    detector = Detector(model_type=model_type)
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    
    if uploaded_video is not None:
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read())  # Save video to disk
    
        st_video = open(vid, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        
        detector.onVideo(vid)
        st_video = open('output.mp4', 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Detected Video") 

def main():

    # Dropdown to select model type
    model_type = st.selectbox(
        'Select Model Type:',
        ('Object Detection (Faster R-CNN)', 'Instance Segmentation (Mask R-CNN)', 'Panoptic Segmentation (Panoptic FPN)')
    )

    # Determine model type for detector initialization and display description
    if model_type == 'Object Detection (Faster R-CNN)':
        model_choice = 'faster_rcnn'
        description = "Faster R-CNN is an object detection model that is good for object detection tasks with speed and accuracy."
    elif model_type == 'Instance Segmenatation (Mask R-CNN)':
        model_choice = 'mask_rcnn'
        description = "Mask R-CNN is an instance segmentation model that provides object detection along with segmentation masks."
    else:
        model_choice = 'panoptic_fpn'  # Default to Panoptic FPN for panoptic segmentation
        description = "Panoptic FPN is a panoptic segmentation model that combines instance segmentation and semantic segmentation for comprehensive scene understanding."

    # Show the description for the selected model
    st.markdown(f"**Description:** {description}")

    option = st.selectbox(
        'Select File Format:',
        ('Images', 'Videos')
    )

    if option == "Images":
        st.subheader('Object Detection and Panoptic Segmentation for Images')
        func_1(model_choice)
    else:
        st.subheader('Object Detection and Panoptic Segmentation for Videos')
        func_2(model_choice)

if __name__ == '__main__':
    main()
