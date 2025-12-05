
def plot_detections(image, detections, output_path=None):
    import cv2
    for det in detections:
        print(f"{det['class_name']}: {det['score']:.3f} at {det['box']}")
        x1, y1, x2, y2 = map(int, det['box'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{det['label']} {det['score']:.2f}", 
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if output_path:
        cv2.imwrite(output_path, image)
    else:
        cv2.imwrite('detected.jpg', image)
    return image