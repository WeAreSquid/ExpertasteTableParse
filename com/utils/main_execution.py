from TableExtractor import TableExtractor
import cv2

#import TableLinesRemover as tlr
#import OcrToTableTool as ottt

image_path = r"./images/primera_foto.jpg"
table_extractor = TableExtractor(image_path)
table_extractor.main_execution()

image_path = r"./images/12_perspective_corrected_with_padding.jpg"
image_to_parse = TableExtractor(image_path, suffix = "_corrected" )
max = image_to_parse.main_execution()
#cv2.imshow("perspective_corrected_image", perspective_corrected_image)

#lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
#image_without_lines = lines_remover.execute()
#cv2.imshow("image_without_lines", image_without_lines)

#ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
#ocr_tool.execute()

#cv2.waitKey(0)
#cv2.destroyAllWindows()