from PIL import Image

def render_from_candidate_and_subset(candidate, subset, H, W) -> Image:
    from controlnet_aux.open_pose import Body as CBody
    from controlnet_aux.open_pose import BodyResult, Keypoint, PoseResult, HWC3, resize_image, draw_poses
    import cv2
    bodies = CBody.format_body_result(candidate, subset)
    results = []
    for body in bodies:
        left_hand, right_hand, face = (None,) * 3
        
        results.append(PoseResult(BodyResult(
            keypoints=[
                Keypoint(
                    x=keypoint.x / float(W),
                    y=keypoint.y / float(H)
                ) if keypoint is not None else None
                for keypoint in body.keypoints
            ], 
            total_score=body.total_score,
            total_parts=body.total_parts
            ), left_hand, right_hand, face))
    

    canvas = draw_poses(results, H, W, draw_body=True, draw_hand=False, draw_face=False) 

    detected_map = canvas
    detected_map = HWC3(detected_map)
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    detected_map = Image.fromarray(detected_map)
    return detected_map