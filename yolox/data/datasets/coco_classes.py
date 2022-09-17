#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

_COCO_CLASSES_AND_COLORS = [
    ["M", [255, 0, 0]],
    ["F", [0, 255, 0]],
    ["N", [0, 0, 255]],
    ["S", [0, 255, 0]],
    ["MASK", [255, 255, 255]],
    ["LF", [255, 255, 0]],
    ["RF", [255, 0, 255]],
    ["BF", [128, 0, 128]],
    ["OF", [128, 255, 128]],
    ["ROT", [125, 0, 102]],
    ["CI", [0, 0, 255]],
    ["OI", [0, 255, 0]],
    ["EB", [255, 255, 255]],
    ["OM", [128, 128, 128]],
    ["CM", [0, 0, 255]],
    ["EH", [0, 0, 255]],
    ["UP", [0, 120, 0]],
    ["UT", [0, 120, 0]],
    ["HOB", [0, 0, 255]],
    ["MOB", [0, 0, 255]],
    ["BK", [0, 0, 255]],
    ["Id", [0, 255, 0]],
    ["SG", [5, 35, 55]],
    ["TL", [200, 80, 100]],
    ["GOG", [5, 35, 55]]
]

COCO_CLASSES = [d[0] for d in _COCO_CLASSES_AND_COLORS]
COCO_CLASSES_VJS = [d[0] for d in _COCO_CLASSES_AND_COLORS]

CLASS_NAMES_FOR_TRAIN_AP = {
    0: "M  ", 
    1: "FE ", 
    2: "NF ", 
    3: "SF ", 
    4: "Fmk", 
    5: "Lff", 
    6: "Rff", 
    7: "Bff", 
    8: "OcF", 
    9: "RtF", 
    10: "CI ", 
    11: "OI ", 
    12: "Eb ", 
    13: "OM ", 
    14: "CM ", 
    15: "EH ", 
    16: "UF ", 
    17: "UT ", 
    18: "HWO", 
    19: "Mob", 
    20: "Bks", 
    21: "Id ", 
    22: "Sgd", 
    23: "T_L", 
    24: "Ntg"
}

# COCO_CLASSES = [
#     "Not_Cat",
#     "Birman", 
#     "British_Shorthair", 
#     "Norwegian_Forest", 
#     "Persian", 
#     "Siamese", 
#     "Sphynx",
#     "Abyssinian",
#     "Scottish_Fold",
#     "Maine_Coon",
#     "Balinese",
#     "Russian_Blue",
#     "Savannah",
#     "Bengal",
#     "British_Longhair",
#     "Burmese",
#     "Dev",
#     "Munchkin",
#     "Oriental_Shorthair",
#     "Ragdoll",
#     "Sai",
#     "Toyger"
# ]


# COCO_CLASSES = [
#     "ID_CARD",
#     "NID_NUMBER",
#     "FIRST_NAME",
#     "SUR_NAME",
#     "DOB",
#     "ISSUE_DATE",
#     "GENDER",
#     "PLACE_BIRTH",
#     "EXPIRY_DATE",
#     "PASSPORT_NO",
#     "OCCUPATION",
#     "VILLAGE_OF_ORIGIN",
#     "CARD_VALUES",
#     "ZIM_PASSPORT",
#     "ZIM_NR",
#     "ZIM_DL",
#     "GAPS"
# ]


# COCO_CLASSES = [
#     "word"
# ]

# COCO_CLASSES = [
#     "person", 
#     "atm_machine", 
#     "helmet", 
#     "shutter"
# ]

# COCO_CLASSES = [
#     "word", 
#     "sign", 
#     "th", 
#     "tc", 
#     "db"
# ]

# CLASS_NAMES_FOR_TRAIN_AP = {
#     idx:d for idx, d in enumerate(COCO_CLASSES)
# }

# COCO_CLASSES = [
#     "CARD_TYPE",
#     "firstName",
#     "lastName",
#     "dateOfBirth", 
#     "placeOfBirth",
#     "village",
#     "district",
#     "chief",
#     "registrationDate",
#     "nrcNumber",
#     "NR_FRONT",
#     "NR_BACK",
#     "idCardNumber",
#     "gender",
#     "dateOfIssue", 
#     "dateOfExpiry", 
#     "placeOfIssue", 
#     "DRIVING_LICENSE",
#     "VOTER_CARD",
#     "PASSPORT",  
#     "VISA",
# ]

# CLASS_NAMES_FOR_TRAIN_AP = {
#     idx:d for idx, d in enumerate(COCO_CLASSES)
# }

# COCO_CLASSES = [d[0] for d in _COCO_CLASSES_AND_COLORS]


COCO_CLASSES = [
    "CARD_TYPE",
    "PAN_NUMBER",
    "AADHAR_NUMBER",
    "NAME ",
    "FATHER_NAME ",
    "DOB ",
    "GENDER",
    "SIGNATURE",
    "MOTHER_NAME",
    "DL_NUMBER",
    "MOBILE_NO",
    "PASSPORT_NO",
    "PLACE_OF_BIRTH",
    "DATE_OF_EXPIRY",
    "DATE_OF_ISSUE",
    "NATIONALITY",
    "SURNAME",
    "TYPE",
    "COUNTRY_CODE",
    "PLACE_OF_ISSUE",
    "PAN CARD",
    "AADHAR CARD",
    "DRIVING LICENSE",
    "PHOTO",
    "FACE",
    "PASSPORT"
]

CLASS_NAMES_FOR_TRAIN_AP = {
    idx:d for idx, d in enumerate(COCO_CLASSES)
}
