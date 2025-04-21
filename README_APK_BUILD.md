# Building the Crop Yield Prediction App for Android

This guide explains how to convert the Streamlit-based Crop Yield Prediction app into an Android APK using Kivy and Buildozer.

## Prerequisites

1. Python 3.7+ installed
2. For Windows users: WSL (Windows Subsystem for Linux) installed
3. For Linux/WSL:
   - Required packages: `build-essential python3-pip python3-dev git`
   - Install with: `sudo apt-get install build-essential python3-pip python3-dev git`

## Setup Steps

### 1. Install Required Python Packages

```bash
pip install kivy==2.1.0 kivymd pillow numpy pandas matplotlib buildozer kivy_garden
# Install matplotlib garden extension
garden install matplotlib
```

### 2. Prepare Your Project Files

Ensure these files are in your project directory:

- `convert_to_apk.py` - The Kivy version of the app
- `buildozer.spec` - Buildozer configuration file
- `static/images/crops/` - Directory for crop images
- `data/crop_info.json` - Crop information data (will be created automatically if missing)

### 3. Building the APK

#### On Linux or WSL:

```bash
# Navigate to your project directory
cd /path/to/project

# Initialize buildozer (if you haven't already)
buildozer init

# Build the debug APK
buildozer -v android debug

# The APK will be in the bin/ directory
```

#### On Windows (using WSL):

1. Open WSL terminal
2. Navigate to your project directory
3. Follow the Linux instructions above

## Custom Crop Images

To add custom crop images to your APK:

1. Create crop images in JPG, JPEG, PNG, or WebP format
2. Name them to match the crop type (e.g., `rice.jpg`, `wheat.png`)
3. Place them in the `static/images/crops/` directory
4. Rebuild the APK

## Troubleshooting

### Common Issues:

1. **Buildozer fails to download dependencies**
   - Solution: Ensure you have a stable internet connection
   - Manual option: Download Android SDK, NDK, and other requirements manually

2. **Build process freezes**
   - Solution: Add `android.accept_sdk_license = True` to your buildozer.spec

3. **Missing libraries during build**
   - Solution: Install the required development libraries:
     ```
     sudo apt-get install -y python3-pip build-essential git python3-dev
     sudo apt-get install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
     sudo apt-get install -y libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev
     sudo apt-get install -y zlib1g-dev
     ```

## Using the App

Once installed on an Android device:

1. Open the app
2. Navigate to the Crop Encyclopedia
3. Select a crop from the dropdown
4. View crop information and images
5. Add your own images using the "Upload Image" button

## Customizing

To add more crops to the app:

1. Edit the `load_crop_info()` function in `convert_to_apk.py`
2. Add new crop entries to the crop_info dictionary
3. Add corresponding images to the `static/images/crops/` directory
4. Rebuild the APK

## Resources

- Kivy documentation: https://kivy.org/doc/stable/
- KivyMD documentation: https://kivymd.readthedocs.io/
- Buildozer documentation: https://buildozer.readthedocs.io/ 