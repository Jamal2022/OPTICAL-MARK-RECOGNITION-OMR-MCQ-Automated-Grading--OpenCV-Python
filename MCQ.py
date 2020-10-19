import cv2
import numpy as np
import utlis
################################
path = "1.jpg"
widthImg = 500
heightImg = 500
question = 5
choise = 5
ans = [1,2,0,1,4]
webcamFed = True
cameraNo = 0
################################

cap = cv2.VideoCapture(cameraNo)
cap.set(10,150)

while True :
    if webcamFed : succes , img = cap.read()
    else :img = cv2.imread(path)

    # PREPROCESSING

    img = cv2.resize(img,(widthImg,heightImg))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBigestContours = img.copy()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,10,50)
    try :
        # Find All Contours
        countours , hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours,countours,-1,(0,255,0),10)

        # Find Rectangle
        rectCon = utlis.rectCountour(countours)
        biggestContours = utlis.getCorneCon(rectCon[0])
        gradePoints = utlis.getCorneCon(rectCon[1])
        #print(biggestContours)

        if biggestContours.size != 0 and gradePoints.size != 0:
            cv2.drawContours(imgBigestContours,biggestContours,-1,(0,255,0),20)
            cv2.drawContours(imgBigestContours,gradePoints,-1,(255,0,0),20)

            biggestContours =  utlis.reorde(biggestContours)
            gradePoints =  utlis.reorde(gradePoints)

            pt1 = np.float32(biggestContours)
            pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1,pt2)
            ImgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            ImgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
            #cv2.imshow("Grade",ImgGradeDisplay)

            # Apply threshold
            ImgWarpGray = cv2.cvtColor(ImgWarpColored,cv2.COLOR_BGR2GRAY)
            ImgThreshold = cv2.threshold(ImgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]

            boxes = utlis.splitBoxes(ImgThreshold)
            #cv2.imshow("Box",boxes[20])


            # Getting No Zero Pixels Values of Each Box
            myPixesVal = np.zeros((question,choise))
            countR = 0
            countC = 0

            for image in boxes:
                totalPixes = cv2.countNonZero(image)
                myPixesVal[countR][countC]=totalPixes
                countC +=1
                if (countC == choise): countR +=1; countC =0
            #print(myPixesVal)

            # FIND INDEX VALUES OF THE MARKING
            myIndex = []
            for x in range(0,question):
                arr = myPixesVal[x]
                #print("ARR",arr)
                myIndexVal = np.where(arr==np.amax(arr))
                #print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])
            #print(myIndex)

            # GRADING
            grading = []
            for x in range(0,question):
                if ans[x]==myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            #print(grading)

            score = (sum(grading)/question) * 100 # FINAL GRADE
            print(score)

            # SHOW ANSWERS
            ImgResult = ImgWarpColored.copy()
            ImgResult = utlis.showAnswers(ImgResult, myIndex, grading, ans, question, choise)
            ImgDrawing = np.zeros_like(ImgWarpColored)
            ImgDrawing = utlis.showAnswers(ImgDrawing, myIndex, grading, ans, question, choise)
            InvMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            ImgInvWarp = cv2.warpPerspective(ImgDrawing, InvMatrix, (widthImg, heightImg))

            ImgRawGrade = np.zeros_like(ImgGradeDisplay)
            cv2.putText(ImgRawGrade,str(int(score))+"%",(50,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,100,255),3)
            #cv2.imshow("Grade Image",ImgRawGrade)
            InvMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            ImgInvGradeDisplay = cv2.warpPerspective(ImgRawGrade, InvMatrixG, (widthImg, heightImg))


            imgFinal = cv2.addWeighted(imgFinal,1,ImgInvWarp,1,0)
            imgFinal = cv2.addWeighted(imgFinal,1,ImgInvGradeDisplay,1,0)

        imgBlank = np.zeros_like(img)
        ImageArray = ([img,imgGray,imgBlur,imgCanny],
                      [imgContours,imgBigestContours,ImgWarpColored,ImgThreshold],
                      [ImgResult,ImgDrawing,ImgInvWarp,imgFinal])
    except :
        imgBlank = np.zeros_like(img)
        ImageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgBlank, imgBlank, imgBlank, imgBlank],
                      [imgBlank, imgBlank, imgBlank, imgBlank])
    lables = [["Original","Gray","Blur","Canny"],
              ["Contours","Biggest Con","Warp","Threshold"],
              ["Result","Raw Drawing","Inv Raw","Final"]]
    ImageStacked = utlis.stackImages(ImageArray,0.3,lables)
    cv2.imshow("The Final Result",imgFinal)
    cv2.imshow("Stacked Images",ImageStacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("FinalImage.jpg",imgFinal)
        cv2.waitKey(300)
        #break