import os
import shutil
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.properties import StringProperty, ListProperty, ObjectProperty
import matplotlib.pyplot as plt
from kivy.clock import Clock
from kivy.utils import platform
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.tab import MDTabsBase
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dialog import MDDialog
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.slider import MDSlider
from kivymd.uix.list import OneLineListItem, MDList
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import io
import json
import threading

# Load crop data - this replaces the CROP_INFO dictionary from the original app
def load_crop_info():
    try:
        with open('data/crop_info.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Create a basic version if file doesn't exist
        crop_info = {
            "rice": {
                "description": "Rice is the seed of the grass species Oryza sativa.",
                "growing_conditions": "Requires warm climate, heavy rainfall, humidity.",
                "nutritional_value": "Good source of carbohydrates, provides some protein.",
                "typical_yield": "4-6 tons per hectare",
                "major_producers": ["China", "India", "Indonesia", "Bangladesh", "Vietnam"]
            },
            "wheat": {
                "description": "Wheat is a grass widely cultivated for its seed.",
                "growing_conditions": "Thrives in well-drained loamy soil.",
                "nutritional_value": "Rich in carbohydrates, with moderate protein.",
                "typical_yield": "3-4 tons per hectare",
                "major_producers": ["China", "India", "Russia", "United States", "France"]
            },
            "maize": {
                "description": "Maize, also known as corn, is a cereal grain.",
                "growing_conditions": "Requires warm weather and plenty of sunshine.",
                "nutritional_value": "High in carbohydrates, provides some protein.",
                "typical_yield": "5-8 tons per hectare",
                "major_producers": ["United States", "China", "Brazil", "Argentina", "Mexico"]
            }
        }
        
        # Save it for future use
        os.makedirs('data', exist_ok=True)
        with open('data/crop_info.json', 'w') as f:
            json.dump(crop_info, f, indent=4)
        
        return crop_info

# Define path for crop images
CROP_IMAGES_PATH = "static/images/crops"

# Create the path if it doesn't exist
os.makedirs(CROP_IMAGES_PATH, exist_ok=True)

# Tab class for MDTabs
class Tab(MDFloatLayout, MDTabsBase):
    """Class implementing content for a tab."""
    pass

# Encyclopedia Tab content
class CropEncyclopediaTab(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.crop_info = load_crop_info()
        self.build_ui()
    
    def build_ui(self):
        layout = MDBoxLayout(orientation='vertical', spacing=10, padding=10, adaptive_height=True)
        
        # Title
        layout.add_widget(MDLabel(
            text="Crop Encyclopedia",
            font_style="H5",
            size_hint_y=None,
            height=50
        ))
        
        # Description
        layout.add_widget(MDLabel(
            text="Explore detailed information about various crops.",
            size_hint_y=None,
            height=30
        ))
        
        # Crop selection spinner
        self.crop_spinner = Spinner(
            text="Select a crop",
            values=list(self.crop_info.keys()),
            size_hint=(1, None),
            height=50
        )
        self.crop_spinner.bind(text=self.on_crop_selected)
        layout.add_widget(self.crop_spinner)
        
        # Info card - will be populated when a crop is selected
        self.info_card = MDCard(
            orientation='vertical',
            size_hint=(1, None),
            height=600,
            padding=20,
            elevation=4
        )
        layout.add_widget(self.info_card)
        
        # Upload new image button
        upload_button = MDRaisedButton(
            text="Upload Image for Selected Crop",
            size_hint=(1, None),
            height=50
        )
        upload_button.bind(on_release=self.show_file_chooser)
        layout.add_widget(upload_button)
        
        self.add_widget(layout)
    
    def on_crop_selected(self, spinner, text):
        if text == "Select a crop" or text not in self.crop_info:
            return
        
        self.info_card.clear_widgets()
        crop_data = self.crop_info[text]
        
        # Create a layout for the card content
        card_layout = MDBoxLayout(orientation='vertical', spacing=10)
        
        # Try to load local image first
        image_layout = MDBoxLayout(
            orientation='vertical',
            size_hint=(1, None),
            height=300
        )
        
        # Check for local image
        extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_found = False
        
        for ext in extensions:
            image_path = os.path.join(CROP_IMAGES_PATH, f"{text}{ext}")
            if os.path.exists(image_path):
                try:
                    img = Image(source=image_path, size_hint=(1, 1))
                    image_layout.add_widget(img)
                    image_found = True
                    break
                except Exception as e:
                    print(f"Error loading image: {e}")
        
        # If no local image found, use a placeholder
        if not image_found:
            placeholder = Image(source='static/images/placeholder.png', size_hint=(1, 1))
            image_layout.add_widget(placeholder)
            image_layout.add_widget(MDLabel(
                text="No image available. Use the Upload button below.",
                halign="center",
                size_hint_y=None,
                height=30
            ))
        
        card_layout.add_widget(image_layout)
        
        # Crop information
        card_layout.add_widget(MDLabel(
            text=text.capitalize(),
            font_style="H5",
            size_hint_y=None,
            height=40
        ))
        
        card_layout.add_widget(MDLabel(
            text=crop_data["description"],
            size_hint_y=None,
            height=60
        ))
        
        card_layout.add_widget(MDLabel(
            text="Growing Conditions:",
            bold=True,
            size_hint_y=None,
            height=30
        ))
        
        card_layout.add_widget(MDLabel(
            text=crop_data["growing_conditions"],
            size_hint_y=None,
            height=60
        ))
        
        card_layout.add_widget(MDLabel(
            text="Nutritional Value:",
            bold=True,
            size_hint_y=None,
            height=30
        ))
        
        card_layout.add_widget(MDLabel(
            text=crop_data["nutritional_value"],
            size_hint_y=None,
            height=60
        ))
        
        card_layout.add_widget(MDLabel(
            text="Typical Yield:",
            bold=True,
            size_hint_y=None,
            height=30
        ))
        
        card_layout.add_widget(MDLabel(
            text=crop_data["typical_yield"],
            size_hint_y=None,
            height=30
        ))
        
        card_layout.add_widget(MDLabel(
            text="Major Producers:",
            bold=True,
            size_hint_y=None,
            height=30
        ))
        
        producers_text = ", ".join(crop_data["major_producers"])
        card_layout.add_widget(MDLabel(
            text=producers_text,
            size_hint_y=None,
            height=30
        ))
        
        self.info_card.add_widget(card_layout)
    
    def show_file_chooser(self, instance):
        if not self.crop_spinner.text or self.crop_spinner.text == "Select a crop":
            MDDialog(
                title="No Crop Selected",
                text="Please select a crop first before uploading an image.",
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: x.parent.parent.dismiss()
                    )
                ]
            ).open()
            return
        
        content = MDBoxLayout(orientation='vertical', spacing=10, padding=20)
        
        if platform == 'android':
            # For Android, use a different approach
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.READ_EXTERNAL_STORAGE])
            
            # We'll use a simpler approach for Android
            content.add_widget(MDLabel(
                text="For Android, we'll open the gallery app"
            ))
            
            dialog = MDDialog(
                title="Upload Image",
                type="custom",
                content_cls=content,
                buttons=[
                    MDFlatButton(
                        text="CANCEL",
                        on_release=lambda x: dialog.dismiss()
                    ),
                    MDRaisedButton(
                        text="OPEN GALLERY",
                        on_release=self.open_android_gallery
                    )
                ]
            )
            dialog.open()
        else:
            # For desktop, use FileChooserListView
            file_chooser = FileChooserListView(
                filters=['*.png', '*.jpg', '*.jpeg', '*.webp'],
                path=os.path.expanduser('~')
            )
            content.add_widget(file_chooser)
            
            dialog = MDDialog(
                title="Choose an Image",
                type="custom",
                content_cls=content,
                size_hint=(0.9, 0.9),
                buttons=[
                    MDFlatButton(
                        text="CANCEL",
                        on_release=lambda x: dialog.dismiss()
                    ),
                    MDRaisedButton(
                        text="SELECT",
                        on_release=lambda x: self.process_selected_file(file_chooser.selection, dialog)
                    )
                ]
            )
            dialog.open()
    
    def open_android_gallery(self, instance):
        # This would use Android intents to open gallery
        # For simplicity, we're just showing a placeholder
        dialog.dismiss()
        
        # Simulate selecting a file
        MDDialog(
            title="Android Gallery",
            text="This would open the Android gallery picker. After selecting an image, it would be processed similar to the desktop version.",
            buttons=[
                MDFlatButton(
                    text="OK",
                    on_release=lambda x: x.parent.parent.dismiss()
                )
            ]
        ).open()
    
    def process_selected_file(self, selection, dialog):
        if not selection:
            return
        
        dialog.dismiss()
        
        try:
            selected_file = selection[0]
            file_ext = os.path.splitext(selected_file)[1].lower()
            
            if file_ext not in ['.jpg', '.jpeg', '.png', '.webp']:
                MDDialog(
                    title="Invalid File",
                    text="Please select a valid image file (JPG, JPEG, PNG, or WebP).",
                    buttons=[
                        MDFlatButton(
                            text="OK",
                            on_release=lambda x: x.parent.parent.dismiss()
                        )
                    ]
                ).open()
                return
            
            # Create destination directory if it doesn't exist
            os.makedirs(CROP_IMAGES_PATH, exist_ok=True)
            
            # Copy the file to our app directory
            crop_name = self.crop_spinner.text.lower()
            destination = os.path.join(CROP_IMAGES_PATH, f"{crop_name}{file_ext}")
            
            # Copy file
            shutil.copy2(selected_file, destination)
            
            # Refresh the display
            self.on_crop_selected(None, self.crop_spinner.text)
            
            MDDialog(
                title="Success",
                text=f"Image saved successfully for {self.crop_spinner.text}.",
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: x.parent.parent.dismiss()
                    )
                ]
            ).open()
            
        except Exception as e:
            MDDialog(
                title="Error",
                text=f"An error occurred: {str(e)}",
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: x.parent.parent.dismiss()
                    )
                ]
            ).open()

# Main App Class
class CropAppKivy(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Green"
        self.theme_cls.accent_palette = "Brown"
        self.theme_cls.primary_hue = "700"
        
        # Main layout
        main_layout = MDBoxLayout(orientation='vertical')
        
        # Toolbar
        toolbar = MDBoxLayout(
            size_hint_y=None,
            height=60,
            md_bg_color=self.theme_cls.primary_color,
            padding=[10, 0, 10, 0]
        )
        
        title = MDLabel(
            text="Crop Yield Prediction",
            font_style="H6",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),
            halign="center"
        )
        
        toolbar.add_widget(title)
        main_layout.add_widget(toolbar)
        
        # Tabs
        self.tabs = MDBoxLayout(orientation='vertical')
        main_layout.add_widget(self.tabs)
        
        # Set up the encyclopedia tab
        self.encyclopedia_tab = CropEncyclopediaTab()
        self.tabs.add_widget(self.encyclopedia_tab)
        
        return main_layout

# This part would be needed for packaging with buildozer for Android
if __name__ == '__main__':
    CropAppKivy().run() 