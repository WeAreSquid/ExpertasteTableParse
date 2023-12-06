import cv2
import numpy as np
from matplotlib import pyplot as plt

class TransformImage():
  def align_images(self):
    # Open the image files.
    img1_color = cv2.imread(r"C:\Users\ferbo\Desktop\GoogleDrive_ferboubeta2\other_projects\TableInImage\ExpertasteTableParse\templates\7_perspective_corrected_primera_foto.jpg")  # Image to be aligned.
    img2_color = cv2.imread(r"C:\Users\ferbo\Desktop\GoogleDrive_ferboubeta2\other_projects\TableInImage\ExpertasteTableParse\templates\crop_initial_template.png")    # Reference image.
  
    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape
      
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(100000)
    
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
      
    # Match features between the two images.
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
      
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
      
    # Sort matches on the basis of their Hamming distance.
    matches = sorted(matches, key = lambda x:x.distance)
      
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
      
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
      
    for i in range(len(matches)):
      p1[i, :] = kp1[matches[i].queryIdx].pt
      p2[i, :] = kp2[matches[i].trainIdx].pt
      
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
      
    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                        homography, (width, height))
      
    # Save the output.
    #cv2.imwrite(r"C:\Users\ferbo\Desktop\GoogleDrive_ferboubeta2\other_projects\TableInImage\ExpertasteTableParse\templates\aligned_output.jpg", transformed_img)

  def align_second(self):
    # Load the template image and the image to be aligned
    #template_image = cv2.imread(r"C:\Users\ferbo\Desktop\GoogleDrive_ferboubeta2\other_projects\TableInImage\ExpertasteTableParse\templates\crop_initial_template.png") 
    #target_image = cv2.imread(r"C:\Users\ferbo\Desktop\GoogleDrive_ferboubeta2\other_projects\TableInImage\ExpertasteTableParse\templates\7_perspective_corrected_primera_foto.jpg")

    # Detect and segment the tables (if needed)

    # Perform feature detection and matching (e.g., using ORB)
    orb = cv2.ORB_create(10000)
    keypoints1, descriptors1 = orb.detectAndCompute(template_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(target_image, None)

    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort the matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Get the corresponding points in the template and target images
    template_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculate the transformation matrix using RANSAC
    transformation_matrix, _ = cv2.estimateAffinePartial2D(target_points, template_points)

    # Apply the transformation to align the target image with the template
    aligned_image = cv2.warpAffine(target_image, transformation_matrix, (template_image.shape[1], template_image.shape[0]))
    
    #cv2.imwrite(r"C:\Users\ferbo\Desktop\GoogleDrive_ferboubeta2\other_projects\TableInImage\ExpertasteTableParse\templates\aligned_output.jpg", aligned_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
  
  def CompleteTable():
    return


if __name__ == "__main__":
    get_cards = TransformImage()
    get_cards.align_second()