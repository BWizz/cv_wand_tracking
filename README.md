# Project Overview
This repo contains a python application and utility scripts in an attempt to re-create the "Wizarding World of Harry Potter" wand tip "spell" detection attraction(s).

Here's examples from the theme park that inspired this project.

https://youtu.be/CeGL_Er046U

## How does it work?
Using a retrofitted night vision security camera, the IR reflection from the wand tip is tracked in real time. The captured motion is normalized and a new image is created that contains the spells character. Using a simple pre-trained neural network designed for character detection, the captured image is validated against a trained data set of expected spells.

If the spell detecting neural network detects a "spell" with a confidence exceeding 80%, then the spell has been classified and printed to the screen. Otherwise, a failed to detect event occurs. 

### State Descriptions
#### Searching
Identifies a stationary wand tip in the image.

#### Waiting
Waits for the wand tip to start moving.

#### Tracking
Captures the location of the wand tip as it moves through the frame.

Note: Tracking is terminated with a stationary wand.

#### Validating
Pre-processes the captured data, feeds it into the spell detector neural network, and prints the results.

Goes back to searching state.

# Required Hardware
1. A compatible IR reflective wand
2. A IR emitting night vision camera retrofitted with a 720nm camera lense
   - I used the following hardware
      - https://www.amazon.com/ELP-1megapixel-Vandal-proof-Industrial-Security-cctv/dp/B00VFLWOC0/ref=sr_1_2?crid=1BI9OFPNOHIEB&keywords=ELP+1+megapixel+Day+Night&qid=1649781050&sprefix=elp+1+megapixel+day+night%2Caps%2C39&sr=8-2
      -  https://www.amazon.com/GREEN-L-IR-Infrared-720nm-Filter/dp/B07PGJM5BR/ref=sr_1_2?crid=3277HU6GLEETK&keywords=GREEN.L%2BInfrared%2Bfilter&qid=1649781155&sprefix=green.l%2Binfrared%2Bfilter%2Caps%2C46&sr=8-2&th=1

## Hardware modifications
The purpose of the hardware modifications is to ensure the camera stays in night mode, even during the day, and filters out everything but IR light.

### Steps
- Cut and attach the 720NM camera lense to fit the night vision security camera that you are using
   - I ended up attaching the camera lense using sticky tack so its not perment.
- If your camera has a seperate light sensor (used to switch between night/day mode), you will need to block it out.
   - Again, I just took the camera apart and covered it with sticky tack. 

## My Setup
![image](https://user-images.githubusercontent.com/28880972/163015825-006c4eb6-776a-4fa2-b820-564cda9a778f.png)

# Running the application
```shell
cd /path/to/repo/cv_wand_tracking/src
python3 wand-tracking.py
```

## Dependencies
- This was developed using python 3 on a Debian OS.
- External Python3 packages used
   - openCV
   - tensorflow

# Current list of supported spells
## Lumos
<img width="122" alt="image" src="https://user-images.githubusercontent.com/28880972/163013372-438f0553-a184-4d72-af99-dbaa17d8b963.png">
https://harrypotter.fandom.com/wiki/Wand-Lighting_Charm

## Wingardium Leviosa
<img width="122" alt="image" src="https://user-images.githubusercontent.com/28880972/163013803-69c5a05a-c90d-4fb4-bb82-b81882645aad.png">
https://harrypotter.fandom.com/wiki/Levitation_Charm

# Training new spells
1. Run `wand-tracking.py` with `recordCaptures = True`. When ran in this mode, all spell characters will be captured
    - Change the save path with `saveLocation = '../../wandCaps/'` if you do not like the default location
2. Perform the same spells over and over again until you have atleast 50-100 captures of the new spell. The more, the better.
3. Create a directory full of spell folders containing your captures for each spell you want to train.
3. In the scripts directory, run the `ml-trainer.py` and modify the data set path to point to your image capture location.
5. You can use the `ml-tester.py` to see how it works on existing captures or
6. Run the `wand-tracking.py` application pointing to your new model and test it out live!
 
# Known Limitations
- If a bright light source is in the spell detector's frame of view, then the application may be thrown off. Its best to use this in-doors without any bright lights in the field of view.
- The current neural network model used is just a prototpye and is in no ways optimized. There might be a better model or approach for this but its a good wrking starting point.
