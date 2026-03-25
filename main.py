import cv2
import numpy as np

# EDIT THESE:
IMAGE_PATH = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEBAQEBESDxMQFRMQEhAQEhIQEBUWFxEZFhYSFRUYHiogGBolGxUYIjEhJSo3Li8uGB8zRDctNyguLi0BCgoKBQUFDgUFDisZExkrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrK//AABEIANYA6wMBIgACEQEDEQH/xAAbAAEAAgMBAQAAAAAAAAAAAAAABQYBAwQCB//EAD8QAAICAgADBgQCCAUBCQAAAAECAAMEEQUSIQYTIjFBURQyQmFxgSMzUmJygpGhFRYkY5JDJTRTVHOjs8HR/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/APuMREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEROHiXGcbG1399VO9AB3VWJPkAPMwO6JAL2sx22Kq8u4j/wAPDyQPyZ0Cn8jMf5hub9Xw3NY/v/DUj+r2/wD1AsESvnI4pb8lONhgj5r7GybAd+Rrr5V8vXnnHxLFFK8+bxi+n7hsXFQbPQBe7JP5kwLZEoVnHvhyr1cSbNXnRfh8iqsPYrMAe4tRE57ADsDrvWvvL7AREQEREBERASqYnabI3ZbbjF8cW30g4osuyamquasd7SBtg3LvaeWx067Frlaub4HMNjdMbOZQ7deWrJChVY/spYqgb8gyj1eB0J2uwdhXu+HJ8hlJZi//AChRJtHDAFSCD1BB2D9wZiytWBVgGB6FWAIP4gyBv7MrUTZw9vgrdluROuJYTslbafLqT8y6b7+kCwzxdaqKWdgiqCWZiFUAeZJPQCV9Mzid/hTHrweXQssySMgk769ylbDmXW9MxHp4T1izs/UAbs/IsyxX+kPxDLXipy9ebuUATQ1vxb1rcCbwM6q+tbabFtRt8roQynR0eo+86JBdjgxx2uZSnxN12SikFWFdlhNfMp+VinKSPcmTsBETntz6UOmtrUj0Z1B/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' # Edit this to change the image path
INACCURACY_VALUE = 0.002; # adjust this to change the accuracy of the approximation. The lower, the more accurate but the more equations. It is recommended not to go over 0.03 (3%), but feel free to experiment.



image = cv2.imread(IMAGE_PATH, 1)
img_height = image.shape[0]

edges = cv2.Canny(image, 100, 200)

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

def bezier_to_equations(start, c1, c2, end, img_height):
    """Convert cubic Bezier curve control points to parametric equations to be represented in Desmos."""

    start = (start[0], img_height - start[1]) # img_height - X inverts y to follow coordinate system.
    c1 = (c1[0], img_height - c1[1])
    c2 = (c2[0], img_height - c2[1])
    end = (end[0], img_height - end[1])

    equation_x = f"((1-t)^3*{start[0]} + 3*(1-t)^2*t*{c1[0]} + 3*(1-t)*t^2*{c2[0]} + t^3*{end[0]})"
    equation_y = f"(1-t)^3*{start[1]} + 3*(1-t)^2*t*{c1[1]} + 3*(1-t)*t^2*{c2[1]} + t^3*{end[1]}"
    return equation_x, equation_y

with open("equations.txt", "w") as f:
    for contour in contours:
        epsilon = INACCURACY_VALUE * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        for i in range(len(approx)):
            start = approx[i][0]
            end = approx[(i+1) % len(approx)][0]
            
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            if dx == 0: # a vertical line
                    f.write(f"x = {start[0]} \\left\\{{{img_height - max(start[1], end[1])} <= y <= {img_height - min(start[1], end[1])}\\right\\}}\n")
            elif dy == 0: # a horizontal line
                f.write(f"y = {img_height - start[1]} \\left\\{{{min(start[0], end[0])} <= x <= {max(start[0], end[0])}\\right\\}}\n")
        
            else: # a curve
                c1 = (2*start[0] + end[0]) / 3, (2*start[1] + end[1]) / 3
                c2 = (start[0] + 2*end[0]) / 3, (start[1] + 2*end[1]) / 3
                
                eq_x, eq_y = bezier_to_equations(start, c1, c2, end, img_height)
                f.write(f"({eq_x}, {eq_y})\n")
