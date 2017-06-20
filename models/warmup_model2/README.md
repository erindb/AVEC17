This is a second warmup task for Desmond's data (see Dropbox for data files).

- feed-forward NN to predict BDI score. BDI scores are 0-36 (note that we had to remove 1 item), so let's just try to predict BDI>18 and BDI<=18.

## data description

Here's what the dataset in Dropbox looks like:

data/selfDisclosure/ 
    AF1M_008.csv
    ...

   9 videos that contain features over time.
   "rating" = participant's self-report valence, from 0-100. This is the dependent variable that you can try to predict.
    * one of the videos has a "flat" rating of all 50...

data/selfDisclosure/pid.csv
   contains a mapping from participantID to video name

data/selfDisclosure/surveys.csv
   contains raw survey data. The BDI might be most relevant.
  - Interpersonal Reactivity Index
  - Berkeley Expressivity Questionnaire
  - Big Five, 10 item (BFI-10)
  - Positive Empathy Scale
  - Interpersonal Regulation Questionnaire
  - Satisfaction with Life Scale
  - Beck Depression Inventory: Short Form. - originally consists of 13 items. IRB made us remove 1 item because it mentions suicidal ideations = 12 items; scored on 4 point Scale, 0,1,2,3, so minimum for whole scale is 0, max is 36.
  - Hypomanic Personality Scale (HPS-20); 20 items, true=1/false=0 scale
  - Toronto Alexithymia Scale (TAS-20); 20 items, 5 points.
  - Social Interaction Anxiety Scale

Desmond also has the .pdf of the surveys in case you wanted to check anything.
