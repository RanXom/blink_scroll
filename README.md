# Eye-Controlled Webpage Scrolling  

A Python-based application that uses webcam eye-tracking to detect blinks and automatically scroll webpages.  
- **Single Blink**: Scrolls down the webpage.  
- **Double Blink**: Scrolls up the webpage.  

## Features  
- Tracks eyes using a webcam and Mediapipe's FaceMesh.  
- Detects single and double blinks for precise control.  
- Simulates keyboard shortcuts for browser scrolling:  
  - `Space` for scrolling down.  
  - `Shift + Space` for scrolling up.  
- Fully standalone and can be converted to an `.exe` for easy usage.  

---

## Requirements  

### Dependencies  
Install the following Python libraries:  
- OpenCV: `pip install opencv-python`  
- Mediapipe: `pip install mediapipe`  
- PyAutoGUI: `pip install pyautogui`  

### Hardware  
- A working webcam for eye tracking.  

---

## Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/eye-controlled-scrolling.git  
   cd eye-controlled-scrolling  
   ```  

2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Run the script:  
   ```bash  
   python eye_scrolling.py  
   ```  

---

## Usage  

1. **Run the Script**  
   Launch the application by running the Python script:  
   ```bash  
   python eye_scrolling.py  
   ```  

2. **Perform Actions**  
   - Blink **once** to scroll **down**.  
   - Blink **twice** (within 1.5 seconds) to scroll **up**.  

3. **Quit the Application**  
   - Press `q` to exit.  

---

## How It Works  

The application leverages Mediapipe's FaceMesh to track the user's eyes in real time.  
1. **Eye Aspect Ratio (EAR):**  
   Calculates EAR to detect when eyes are closed (blink).  

2. **Blink Detection:**  
   - Counts blinks within a defined time window to distinguish between single and double blinks.  

3. **Keyboard Simulation:**  
   Uses PyAutoGUI to send keypress events (`Space` and `Shift + Space`) to the browser.  
