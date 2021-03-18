-------------------------------------------------------------------------------
README file for Task files for the REVERB challenge

!! Read This document before using the task files !!

For any inquiries regarding this document, please contact us via 
REVERB-challenge@lab.ntt.co.jp
-------------------------------------------------------------------------------

This document explains:

1. The naming convention of the task files
2. Which task files to use
3. Directory structure

---
1. Naming convention for the task files

  To use the task files correctly, it is first important to understand
  the naming convention of them. 

  The name of the task files indicates for which settings it is for:
  - Dataset: SimData (i.e. REVERB_WSJCAM0 data) or RealData (i.e. MC_WSJ_AV data)
  - Testset: development test set (dt), evaluation test set (et) or multi-condition training data (tr)
  - Recording type: 1ch, 2ch, 8ch noisy reverberant recording, or reference clean recording
  - Position: near or far
  - Room type: room1, room2 or room3 
       For SimData:room1 corresponds to a small-size room
                   room2 corresponds to a medium-size room
                   room3 corresponds to a large-size room
       For RealData: there is only room1 which corresponds to a large-size room
  - Microphone index: 1st (i.e., _A), 2nd (i.e., _B), 3rd, 4th, 5th, 6th, 7th or 8th (i.e., _H) microphones
  
  Consequently, the name of the task file become for example as follows. 
  ex: RealData_dt_for_1ch_far_room1_A or SimData_dt_for_cln_room1

  And their variation can be summarized as follows.
  (RealData,SimData)_(dt,et,tr)_for_(1ch,2ch,8ch,cln)_(near,far)_room(1,2,3)_(A,B,...,H)

  Note that the variables in parentheses indicate the full variations that task files 
  can take.


2. Which task files to use

   First, the files listed in the task files with the name of *_A should be 
   considered as a set of speech signals captured at a lead/reference microphone.
   This rule is meant to be common for all the participants.
   !! Please do not change the lead/reference microphone !!

   -For those who would like to evaluate algorithms with a 1ch input
     
     Please use the files under the directory taskFiles/1ch/.
     You may find the task files with the following names.

     (1) SimData_dt_for_cln_room(1,2,3)
     (2) SimData_dt_for_1ch_(near,far)_room(1,2,3)_A
     (3) RealData_dt_for_1ch_(near,far)_room1_A
     (4) SimData_tr_for_1ch_A

   - For those who would like to evaluate algorithms with 2ch input
     Please use the files under the directory taskFiles/2ch/.
     You may find the task files with the following names.

     (1) SimData_dt_for_cln_room(1,2,3)
     (2) SimData_dt_for_2ch_(near,far)_room(1,2,3)_(A,B)
     (3) RealData_dt_for_2ch_(near,far)_room1_(A,B)
     (4) SimData_tr_for_2ch_(A,B)

   - For those who would like to evaluate algorithms with 8ch input
     Please use the files under the directory taskFiles/8ch/.
     You may find the task files with the following names.

     (1) SimData_dt_for_cln_room(1,2,3)
     (2) SimData_dt_for_8ch_(near,far)_room(1,2,3)_(A,B,C,D,E,F,G,H)
     (3) RealData_dt_for_8ch_(near,far)_room1_(A,B,C,D,E,F,G,H)
     (4) SimData_tr_for_8ch_(A,B,C,D,E,F,G,H)

   The first set of task files (1) should be used to compare observed/processed speech signals
   with clean speech signals. Entries in these files should not be used for processing 
   observed signal in any way.

   The second  set of task files (2) should be used to load noisy reverberant speech signals of a certain
   test condition (position, room type) of SimData. You can specify the condition by appropriately choosing 
   the variables in parentheses. The set of task files ending with "_A" should be considered 
   as observation at a reference microphone, while those in task files ending with "_B" to "_H" 
   should be assigned to 2nd to 8th channels respectively.
   Resultant processed signals based on the files should be generated such that
   they are time-aligned with the observed reverberant speech signals in the task files ending with "_A", 
   because the SE evaluation tool works based on this assumption. 
     
   The third set of task files (3) should be used to load noisy reverberant speech signals of a certain
   test condition (position) of RealData. You can specify the condition by appropriately choosing 
   the variables in parentheses.

   The fourth set of task files (4) should be used to load multi-condition training data.
 		
   Note that each line in the task files corresponds to an utterance, and
   files in the same line in *_A to *_H indicate the same utterance captured at different
   microphones. 

3. Directory structure

   The following is the directory structure of the task files and all the files.
   Note that the files for evaluation dataset (ones with the name of *_et_*) are not contained
   in the initial distribution. They will be distributed on the release date of the evaluation dataset.

|-- 1ch
|   |-- RealData_dt_for_1ch_far_room1_A
|   |-- RealData_dt_for_1ch_near_room1_A
|   |-- RealData_et_for_1ch_far_room1_A
|   |-- RealData_et_for_1ch_near_room1_A
|   |-- SimData_dt_for_1ch_far_room1_A
|   |-- SimData_dt_for_1ch_far_room2_A
|   |-- SimData_dt_for_1ch_far_room3_A
|   |-- SimData_dt_for_1ch_near_room1_A
|   |-- SimData_dt_for_1ch_near_room2_A
|   |-- SimData_dt_for_1ch_near_room3_A
|   |-- SimData_dt_for_cln_room1
|   |-- SimData_dt_for_cln_room2
|   |-- SimData_dt_for_cln_room3
|   |-- SimData_et_for_1ch_far_room1_A
|   |-- SimData_et_for_1ch_far_room2_A
|   |-- SimData_et_for_1ch_far_room3_A
|   |-- SimData_et_for_1ch_near_room1_A
|   |-- SimData_et_for_1ch_near_room2_A
|   |-- SimData_et_for_1ch_near_room3_A
|   |-- SimData_et_for_cln_room1
|   |-- SimData_et_for_cln_room2
|   |-- SimData_et_for_cln_room3
|   `-- SimData_tr_for_1ch_A
|-- 2ch
|   |-- RealData_dt_for_2ch_far_room1_A
|   |-- RealData_dt_for_2ch_far_room1_B
|   |-- RealData_dt_for_2ch_near_room1_A
|   |-- RealData_dt_for_2ch_near_room1_B
|   |-- RealData_et_for_2ch_far_room1_A
|   |-- RealData_et_for_2ch_far_room1_B
|   |-- RealData_et_for_2ch_near_room1_A
|   |-- RealData_et_for_2ch_near_room1_B
|   |-- SimData_dt_for_2ch_far_room1_A
|   |-- SimData_dt_for_2ch_far_room1_B
|   |-- SimData_dt_for_2ch_far_room2_A
|   |-- SimData_dt_for_2ch_far_room2_B
|   |-- SimData_dt_for_2ch_far_room3_A
|   |-- SimData_dt_for_2ch_far_room3_B
|   |-- SimData_dt_for_2ch_near_room1_A
|   |-- SimData_dt_for_2ch_near_room1_B
|   |-- SimData_dt_for_2ch_near_room2_A
|   |-- SimData_dt_for_2ch_near_room2_B
|   |-- SimData_dt_for_2ch_near_room3_A
|   |-- SimData_dt_for_2ch_near_room3_B
|   |-- SimData_dt_for_cln_room1
|   |-- SimData_dt_for_cln_room2
|   |-- SimData_dt_for_cln_room3
|   |-- SimData_et_for_2ch_far_room1_A
|   |-- SimData_et_for_2ch_far_room1_B
|   |-- SimData_et_for_2ch_far_room2_A
|   |-- SimData_et_for_2ch_far_room2_B
|   |-- SimData_et_for_2ch_far_room3_A
|   |-- SimData_et_for_2ch_far_room3_B
|   |-- SimData_et_for_2ch_near_room1_A
|   |-- SimData_et_for_2ch_near_room1_B
|   |-- SimData_et_for_2ch_near_room2_A
|   |-- SimData_et_for_2ch_near_room2_B
|   |-- SimData_et_for_2ch_near_room3_A
|   |-- SimData_et_for_2ch_near_room3_B
|   |-- SimData_et_for_cln_room1
|   |-- SimData_et_for_cln_room2
|   |-- SimData_et_for_cln_room3
|   |-- SimData_tr_for_2ch_A
|   `-- SimData_tr_for_2ch_B
`-- 8ch
    |-- RealData_dt_for_8ch_far_room1_A
    |-- RealData_dt_for_8ch_far_room1_B
    |-- RealData_dt_for_8ch_far_room1_C
    |-- RealData_dt_for_8ch_far_room1_D
    |-- RealData_dt_for_8ch_far_room1_E
    |-- RealData_dt_for_8ch_far_room1_F
    |-- RealData_dt_for_8ch_far_room1_G
    |-- RealData_dt_for_8ch_far_room1_H
    |-- RealData_dt_for_8ch_near_room1_A
    |-- RealData_dt_for_8ch_near_room1_B
    |-- RealData_dt_for_8ch_near_room1_C
    |-- RealData_dt_for_8ch_near_room1_D
    |-- RealData_dt_for_8ch_near_room1_E
    |-- RealData_dt_for_8ch_near_room1_F
    |-- RealData_dt_for_8ch_near_room1_G
    |-- RealData_dt_for_8ch_near_room1_H
    |-- RealData_et_for_8ch_far_room1_A
    |-- RealData_et_for_8ch_far_room1_B
    |-- RealData_et_for_8ch_far_room1_C
    |-- RealData_et_for_8ch_far_room1_D
    |-- RealData_et_for_8ch_far_room1_E
    |-- RealData_et_for_8ch_far_room1_F
    |-- RealData_et_for_8ch_far_room1_G
    |-- RealData_et_for_8ch_far_room1_H
    |-- RealData_et_for_8ch_near_room1_A
    |-- RealData_et_for_8ch_near_room1_B
    |-- RealData_et_for_8ch_near_room1_C
    |-- RealData_et_for_8ch_near_room1_D
    |-- RealData_et_for_8ch_near_room1_E
    |-- RealData_et_for_8ch_near_room1_F
    |-- RealData_et_for_8ch_near_room1_G
    |-- RealData_et_for_8ch_near_room1_H
    |-- SimData_dt_for_8ch_far_room1_A
    |-- SimData_dt_for_8ch_far_room1_B
    |-- SimData_dt_for_8ch_far_room1_C
    |-- SimData_dt_for_8ch_far_room1_D
    |-- SimData_dt_for_8ch_far_room1_E
    |-- SimData_dt_for_8ch_far_room1_F
    |-- SimData_dt_for_8ch_far_room1_G
    |-- SimData_dt_for_8ch_far_room1_H
    |-- SimData_dt_for_8ch_far_room2_A
    |-- SimData_dt_for_8ch_far_room2_B
    |-- SimData_dt_for_8ch_far_room2_C
    |-- SimData_dt_for_8ch_far_room2_D
    |-- SimData_dt_for_8ch_far_room2_E
    |-- SimData_dt_for_8ch_far_room2_F
    |-- SimData_dt_for_8ch_far_room2_G
    |-- SimData_dt_for_8ch_far_room2_H
    |-- SimData_dt_for_8ch_far_room3_A
    |-- SimData_dt_for_8ch_far_room3_B
    |-- SimData_dt_for_8ch_far_room3_C
    |-- SimData_dt_for_8ch_far_room3_D
    |-- SimData_dt_for_8ch_far_room3_E
    |-- SimData_dt_for_8ch_far_room3_F
    |-- SimData_dt_for_8ch_far_room3_G
    |-- SimData_dt_for_8ch_far_room3_H
    |-- SimData_dt_for_8ch_near_room1_A
    |-- SimData_dt_for_8ch_near_room1_B
    |-- SimData_dt_for_8ch_near_room1_C
    |-- SimData_dt_for_8ch_near_room1_D
    |-- SimData_dt_for_8ch_near_room1_E
    |-- SimData_dt_for_8ch_near_room1_F
    |-- SimData_dt_for_8ch_near_room1_G
    |-- SimData_dt_for_8ch_near_room1_H
    |-- SimData_dt_for_8ch_near_room2_A
    |-- SimData_dt_for_8ch_near_room2_B
    |-- SimData_dt_for_8ch_near_room2_C
    |-- SimData_dt_for_8ch_near_room2_D
    |-- SimData_dt_for_8ch_near_room2_E
    |-- SimData_dt_for_8ch_near_room2_F
    |-- SimData_dt_for_8ch_near_room2_G
    |-- SimData_dt_for_8ch_near_room2_H
    |-- SimData_dt_for_8ch_near_room3_A
    |-- SimData_dt_for_8ch_near_room3_B
    |-- SimData_dt_for_8ch_near_room3_C
    |-- SimData_dt_for_8ch_near_room3_D
    |-- SimData_dt_for_8ch_near_room3_E
    |-- SimData_dt_for_8ch_near_room3_F
    |-- SimData_dt_for_8ch_near_room3_G
    |-- SimData_dt_for_8ch_near_room3_H
    |-- SimData_dt_for_cln_room1
    |-- SimData_dt_for_cln_room2
    |-- SimData_dt_for_cln_room3
    |-- SimData_et_for_8ch_far_room1_A
    |-- SimData_et_for_8ch_far_room1_B
    |-- SimData_et_for_8ch_far_room1_C
    |-- SimData_et_for_8ch_far_room1_D
    |-- SimData_et_for_8ch_far_room1_E
    |-- SimData_et_for_8ch_far_room1_F
    |-- SimData_et_for_8ch_far_room1_G
    |-- SimData_et_for_8ch_far_room1_H
    |-- SimData_et_for_8ch_far_room2_A
    |-- SimData_et_for_8ch_far_room2_B
    |-- SimData_et_for_8ch_far_room2_C
    |-- SimData_et_for_8ch_far_room2_D
    |-- SimData_et_for_8ch_far_room2_E
    |-- SimData_et_for_8ch_far_room2_F
    |-- SimData_et_for_8ch_far_room2_G
    |-- SimData_et_for_8ch_far_room2_H
    |-- SimData_et_for_8ch_far_room3_A
    |-- SimData_et_for_8ch_far_room3_B
    |-- SimData_et_for_8ch_far_room3_C
    |-- SimData_et_for_8ch_far_room3_D
    |-- SimData_et_for_8ch_far_room3_E
    |-- SimData_et_for_8ch_far_room3_F
    |-- SimData_et_for_8ch_far_room3_G
    |-- SimData_et_for_8ch_far_room3_H
    |-- SimData_et_for_8ch_near_room1_A
    |-- SimData_et_for_8ch_near_room1_B
    |-- SimData_et_for_8ch_near_room1_C
    |-- SimData_et_for_8ch_near_room1_D
    |-- SimData_et_for_8ch_near_room1_E
    |-- SimData_et_for_8ch_near_room1_F
    |-- SimData_et_for_8ch_near_room1_G
    |-- SimData_et_for_8ch_near_room1_H
    |-- SimData_et_for_8ch_near_room2_A
    |-- SimData_et_for_8ch_near_room2_B
    |-- SimData_et_for_8ch_near_room2_C
    |-- SimData_et_for_8ch_near_room2_D
    |-- SimData_et_for_8ch_near_room2_E
    |-- SimData_et_for_8ch_near_room2_F
    |-- SimData_et_for_8ch_near_room2_G
    |-- SimData_et_for_8ch_near_room2_H
    |-- SimData_et_for_8ch_near_room3_A
    |-- SimData_et_for_8ch_near_room3_B
    |-- SimData_et_for_8ch_near_room3_C
    |-- SimData_et_for_8ch_near_room3_D
    |-- SimData_et_for_8ch_near_room3_E
    |-- SimData_et_for_8ch_near_room3_F
    |-- SimData_et_for_8ch_near_room3_G
    |-- SimData_et_for_8ch_near_room3_H
    |-- SimData_et_for_cln_room1
    |-- SimData_et_for_cln_room2
    |-- SimData_et_for_cln_room3
    |-- SimData_tr_for_8ch_A
    |-- SimData_tr_for_8ch_B
    |-- SimData_tr_for_8ch_C
    |-- SimData_tr_for_8ch_D
    |-- SimData_tr_for_8ch_E
    |-- SimData_tr_for_8ch_F
    |-- SimData_tr_for_8ch_G
    `-- SimData_tr_for_8ch_H

