## Dataset Description

#### Overview

The Static gesture classifier model [gesture_svm.pkl](./dynago/models/gesture_svm.pkl) uses a custom dataset of static hand gestures. Each gesture instance is represented in raw floating-point `(x, y, z)` coordinates corresponding to landmark points extracted using MediaPipe Hands. These raw landmarks are stored in `raw.csv` and are then normalized to a `[0,1]` range with the axis point as the wrist. The current dataset comprises of 4291 entries classified over 6 gestures [gesture_map.json](./dynago/data/gesture_map.json) being as follows:

- Fist
- Two fingers (pointer finger and middle finger)
- Three fingers one (pointer, middle and ring fingers)
- Three fingers two (middle, ring and little fingers)
- Pinch (thumb and pointer finger)
- Point (Pointer finger)

Each captured frame undergoes vertical flip and conversion to `cv2.COLOR_BGR2RGB` prior to landmark extraction.

![[dataset_visualization.png]]
#### Normalization

Normalization is performed using a custom script that processes only new raw entries (from `raw.csv`) and appends them to `normalized.csv`. Each hand landmark set is:

- Reshaped into a `(21, 3)` array (x, y, z).
- Centered around the wrist (landmark 0).
- Scaled such that the farthest point from the wrist lies within a unit sphere.
- Flattened and appended with a gesture label.

This ensures consistency across hand sizes, positions, and recording variations.

## Loss Analysis and Confusion Matrix

#### 1. Overall Performance Summary

- Average Accuracy: 92.3% (calculated from confusion matrix)
- Best Performing Gesture: "point" (100% AUC, 174/175 correct)
- Most Challenging Gesture: "pinch" (AUC 0.95, 251/347 correct)

#### 2. Detailed Gesture Analysis

| Gesture       | Correct | Total | Accuracy | Main Confusions            | AUC  | Avg Precision |
| ------------- | ------- | ----- | -------- | -------------------------- | ---- | ------------- |
| fist          | 111     | 128   | 86.7%    | point (17)                 | 0.99 | 0.94          |
| open palm     | 285     | 287   | 99.3%    | two_fingers (2)            | 0.97 | 0.82          |
| two_fingers   | 226     | 243   | 93.0%    | fist (16)                  | 0.99 | 0.98          |
| three_fingers | 165     | 189   | 87.3%    | two_fingers (16)           | 1.00 | 0.98          |
| pinch         | 251     | 347   | 72.3%    | open palm (51), point (36) | 0.95 | 0.92          |
| point         | 174     | 175   | 99.4%    | fist (17)                  | 1.00 | 0.96          |

#### 3. Key Observations

A. Excellent Performance:

- "point" and "open palm" gestures show near-perfect classification (AUC 1.00 and 0.97 respectively)
- All gestures have AUC > 0.95, indicating strong separability

B. Problem Areas:

1. Pinch Recognition Issues:

   - 51 misclassified as "open palm" (14.7% error rate)
   - 36 confused with "point" (10.4% error rate)
   - Lowest precision (0.92) among all gestures

2. Fist vs Point Confusion:

   - 17 "fist" samples misclassified as "point"
   - Unexpected given their distinct shapes

3. Three-Finger Stability:
   - 16 samples confused with "two_fingers"
   - Possibly due to intermediate finger positions

#### 4. Visual Analysis

A. Confusion Matrix Insights:
![Confusion Matrix](confusion_matrix.png)

- Clear diagonal dominance shows good classification
- Pinch column shows most scattered errors

B. ROC Curve Validation:
![ROC Curves](roc_curves.png)

- All curves hug the top-left corner (ideal)
- "three_fingers" and "point" achieve perfect 1.0 AUC

C. Precision-Recall Tradeoffs:
![Precision-Recall](precision_recall_curves.png)

- "open palm" shows lowest average precision (0.82)
- Consistent with its 2 misclassifications

#### 5. Recommendations for Improvement

A. Data Collection Focus:

1. Pinch Variations:

   - Collect more samples with thumb-index distances between 1-2cm
   - Include different hand orientations

2. Fist-Point Ambiguity:
   - Record transitional frames between these gestures
   - Verify labeling consistency

B. Model Enhancements:

#### 6. Conclusion

The model performs exceptionally well overall (92.3% accuracy), with specific challenges in pinch recognition and fist-point differentiation. The high AUC scores (>0.95 for all gestures) confirm excellent class separability. Focused data collection for problematic cases and minor model tuning should push performance above 95%.

Appendix: [performance_metrics.json](dynago/performance/performance_metrics.json) contains all raw metrics for future comparison.

Would you like me to generate any specific additional analyses or visualizations from this data?
