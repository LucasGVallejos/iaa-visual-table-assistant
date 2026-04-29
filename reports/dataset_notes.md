# Dataset Notes

## Overview

This document tracks dataset sources, selected classes, preprocessing decisions and known issues for the visual table assistant project.

The prepared dataset will be built from public datasets and converted into YOLO format.

## Selected Sources

- Open Images V7: general everyday objects.
- OCID: cluttered tabletop scenes and object-level annotations.
- UEC FOOD-256: food detection with bounding boxes.

## Initial Classes

| ID | Class | Description |
|---:|-------|-------------|
| 0 | food | Generic visible food item. Specific food type is not identified. |
| 1 | cup_glass | Cups, glasses or mugs. |
| 2 | bottle | Bottles or similar vertical liquid containers. |
| 3 | plate_bowl | Plates, bowls or food containers. |
| 4 | spoon | Spoon used as tableware. |
| 5 | fork | Fork used as tableware. |
| 6 | knife | Knife used as tableware or cutting utensil. |

## Preprocessing Decisions

- TBD: class filtering strategy.
- TBD: label mapping between dataset-specific labels and project classes.
- TBD: conversion to YOLO format.
- TBD: train/validation/test split.
- TBD: class balancing strategy.

## Known Issues

- TBD

## Class Distribution

| Class | Train | Val | Test |
|-------|------:|----:|-----:|
| food | - | - | - |
| cup_glass | - | - | - |
| bottle | - | - | - |
| plate_bowl | - | - | - |
| spoon | - | - | - |
| fork | - | - | - |
| knife | - | - | - |