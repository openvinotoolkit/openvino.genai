from PIL import Image

def debug_print(*args, **kwargs):
    import os
    if os.getenv("DEBUG"):
        print(*args, **kwargs)


def render_from_candidate_and_subset(candidate, subset, H, W) -> Image:
    from controlnet_aux.open_pose import Body as CBody
    from controlnet_aux.open_pose import BodyResult, Keypoint, PoseResult, HWC3, resize_image, draw_poses
    import cv2
    bodies = CBody.format_body_result(candidate, subset)
    debug_print(bodies)

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
    
    debug_print(results)

    canvas = draw_poses(results, H, W, draw_body=True, draw_hand=False, draw_face=False) 

    detected_map = canvas
    detected_map = HWC3(detected_map)
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    detected_map = Image.fromarray(detected_map)
    return detected_map


def cv_gaussian_blur(image, sigma):
    import cv2
    truncate = 4
    radius = int(truncate * sigma + 0.5)
    ksize = 2 * radius + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)