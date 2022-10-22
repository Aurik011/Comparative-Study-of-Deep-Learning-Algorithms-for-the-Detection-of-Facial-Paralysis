
# CSE498R-Directed-Research

Comparative Study of Deep Learning Algorithms for the Detection of Facial Paralysis

# Publication 
Syed Maaher Hossain, Zeeshan Jamal, Aurik Anjum Noshin and Mohammad Monirujjaman Khan, “Comparative Study of Deep Learning Algorithms for the Detection of Facial Paralysis,” 2022 IEEE 12th Annual Information Technology, Electronics and Mobile Communication Conference (IEEE IEMCON), 12-15 October, Vancouver, Canada. Scopus Indexed. Accepted and due for publication., 2022

# ABSTRACT
Statistical analysis has shown a significant increase in the rate of brain strokes in recent times. But the time required to carry out the conventional methods to detect brain stroke on a patient requires both trained personnel and time. Any discrepancy in the process might also lead to fatal consequences. One of the most common symptoms of stroke includes facial paralysis. Currently, there is a lack of efficient models for detecting facial paralysis. To help make the stroke detection technique less formidable to at least a certain extent, this report presents a comparative analysis of deep learning algorithms for the efficient detection of facial paralysis. We have compared different pre-trained models to find the best suitable for detecting facial paralysis. In our research, we successfully ran tests on datasets consisting of facial images. We were able to attain the highest training accuracy and validation accuracy using the VGG-16 deep learning model, which was 99.67% and 99.37%, respectively.

# INTRODUCTION
In Bangladesh itself, stroke is the third leading cause of death, where compared to 2015, the death toll due to stroke increased by 13.7% in 2020. [1] Half of the stroke survivors face chronic disability. Damage caused by stroke may reduce if diagnosed early. One of the most common symptoms of stroke is facial paralysis. The main goal of our project is to find out which deep learning algorithm, among the three we compared, is the best for the detection of facial paralysis. We compared a total of three different deep learning models. As datasets for creating the models, we used images that consist of images of both stroke patients’ faces and normal faces. The three deep learning models used were VGG16, InceptionV3, and ResNet50. All three of these models showed the potential to be efficient for the detection of facial paralysis. It seems VGG16 is more effective in certain aspects compared to the other two deep learning models.

# METHODOLOGY
In this chapter, the methodology and structure of our project are discussed in detail. Stroke is the fifth leading cause of death in the world. A stroke happens when the blood flow to the brain is interrupted. [4] Death of brain cells and permanent brain damage can occur if oxygen-rich blood does not reach your brain. There are mainly two types of brain strokes.
1.      Ischemic Stroke: - A blood clot blocks the flow of blood to the brain.
2.      Hemorrhagic Stroke: - The bursting of a weak blood vessel causes internal bleeding.
A stroke can cause temporary or permanent disability, depending on the severity or for how long blood flow was interrupted.  There are many survivors of stroke, who recover with rehabilitation such as occupational, speech, or physical therapy. But, the sooner the signs of a stroke are recognized and medical help is taken, the better the chances of recovering and avoiding serious brain damage or disability.

# THE FAST METHOD 
A person having a stroke may experience one or both of the above-mentioned symptoms, but a patient may not realize that he/she has serious problems until it is too late. Stroke symptoms can develop slowly over hours or days. Ministrokes or Transient Ischemic Attack (TIA) symptoms are temporary and usually improve within hours. But these can increase the risk of an actual stroke, and these signs should never be ignored. Any signs or symptoms of stroke require further investigation by doctors. If a patient can get to the hospital within three hours after the first symptoms of an ischemic stroke, doctors can give medication to dissolve blood clots and restore blood flow to the brain. Fast action can improve the chances of recovering fully from a stroke, and also reduce the chances of having another stroke in the future. As a result, testing is very important.
A simple test using the FAST method can help identify stroke
1. Face: - Ask the person to smile. Look for signs of drooping on either side of the face.
2. Arms: - Ask the person to raise their arms. Look for a downward drift in any one arm
3. Speech: - Ask the person to repeat a phrase without slurring
4. Time: - No time should be wasted. Local emergency services should be immediately called if a person shows signs of a stroke.

The scanning of the face can help detect if the user has any signs of facial drooping, which is a major symptom of brain stroke.

# APPROACH
We used the pre-trained models VGG-16, InceptionV3 and ResNet50 in our project for the detection of facial paralysis. The reason behind using these is the fact that VGG-16, InceptionV3 and Resnet50 are very powerful pre-trained models, and they have shown great accuracy on image classification in past works. If we didn't use pre-trained models, we may have faced difficulties in the weight initialization and accuracy values of our model. We used Python’s Tensorflow & Keras framework to implement the model. Input images were vectorized and passed through these models. Finally, concatenated outputs were sent through the softmax activation function for classification.

# DATASET 
Deep learning models utilize the image files for analysis, which requires datasets that contain clear and reliable information. Therefore, we have selected two datasets consisting of very clear images for our project. One consists of paralyzed face images, and the other, normal face images.  For the drooped face images, we have chosen The Facial_Droop_and_Facial_Paralysis_image [5] dataset, which contains 1024 image files. For the normal face images, we have chosen the UTK Face Cropped[6] dataset, which contains around 23,700 image files. During data labeling, we have taken around 800 stroke face images and 1000 normal face images for training purposes. For the validation set, we wanted to achieve an 80-20 ratio so we labeled 200 stroke faces and 200 normal faces here.


# RELATED WORKS
In the paper “Facial Paralysis Detection on Images Using Key Point Analysis”, a method used to detect facial paralysis from the photograph of a person’s face has been proposed [2]. The design of the system included three modules- facial landmark extraction, facial measure computation, and facial paralysis classification. The facial measure module aimed to identify asymmetry levels within the face using facial landmarks, and the output label was provided by a binary classifier based on a multi-layer perceptron approach. The tests yielded outstanding classification results on images.
       A novel approach for the quantitative assessment of facial paralysis that deals with classification problems for FP type and degree of severity has been introduced in the paper titled “Efficient quantitative assessment of facial paralysis using iris segmentation and active contour-based key points detection with hybrid classifier”. An algorithm that extracts the human iris, and detects facial landmarks, and a hybrid-approach combining the rule-based and machine learning algorithm to analyze facial paralysis using the captured images is presented here [3].
       The paper titled “Application of Machine Learning for Facial Stroke Detection” proposes a facial stroke recognition system that assists patients in self-detection. Facial landmarks were tracked by an ensemble of regression tree methods (ERT). To classify stroke, the Support Vector Machine (SVM), Random Forest (RF), and Bayesian Classifier were used as classifiers. The results showed that the system could accurately and effectively distinguish stroke from facial images with accuracies of 100%, 95.45% and 100% respectively [4].
       An efficient yet objective facial paralysis assessment approach via automatic computational image analysis is proposed in the paper “Automatic Facial Paralysis Assessment via Computational Image Analysis”. The proposed method was validated on 80 FP patients, and quantitative results demonstrated that the method, which achieved an accuracy of 97.14%, outperformed many state-of-the-art systems [5].
      A study which preprocesses data to improve the image quality of CT scans of stroke patients by optimizing the quality of image results and to reduce noise, and also applies machine learning algorithms to classify the images into two sub-types of strokes is presented in the paper “Machine Learning Algorithms for Stroke Disease Classification” [6]. The Random Forest algorithm generated the highest level of accuracy (95.97%), along with a precision of 94.39%, recall of 96.12% and f-1 measure of 95.39%.
      Studies based on early stroke detection are mostly focused on different scanning processes such as
Retinal scanning or MRI or CT scanning. But these systems are not available for masses of people and are only used by doctors for treatment purposes. However, one of the studies (StrokeSave: A Novel, High-Performance Mobile Application for Stroke Diagnosis using Deep Learning and Computer Vision) used a heart rate and blood pressure monitoring system with a wrist machine to predict the chance of stroke [7].
       The paper, “Retina as a biomarker of stroke” gives us the idea of using retinal scan as an option to predict stroke early [8]. Their research was based on the idea of considering retinal features such as fractal dimension, asymmetry factor to foresee the risk of stroke.
        A study on the FAST method using the deep learning model has been shown in the paper Prehospital stroke scale (FAST PLUS Test) predicts patients with intracranial large vessel occlusion [9]. Their technology was focused on dividing the FAST method into two major parts as FAST PLUS test 1 and
FAST PLUS test 2. 
       The paper “A multi-path 2.5-dimensional convolutional neural network system for segmenting stroke lesions in brain MRI images” proposes a multi-modal, multi-path convolutional neural network system for the automation of stroke lesion segmentation [10]. The system makes use of nine end-to-end UNets that take as input 2D slices and examines all three planes with three different normalizations. Outputs from the nine paths are concatenated into a 3D volume, which is in turn passed to a 3D CNN to get a final lesion mask as output. The model was trained and tested on datasets from a total of three sources. The results achieved using cross-study validation were compared with previous methods based on naive Bayes, random forests and three recently published CNNs, and showed superior performance compared to them.
       When dealing with stroke, it is always important to know what we want to identify. We use the FAST (Face, Arm, Speech, Time) test to check the most common symptoms of a stroke. Face test is the first and foremost one in which a patient has to smile to check whether one side of the face is drooping [11]. The droopy face is also a sign of brain and nerve disorder known as Bell's palsy, that is, muscle weakness or paralysis on one side of the neck. Therefore, the facial nerve damage that controls the muscles on one side of the face causes the other side of the face to droop [12].
Chin et al. developed an automated early ischemic stroke detection system. In this research, they have utilized Convolutional Neural Network (CNN), a deep learning algorithm on the dataset of CT images of the brain. Experiment results have shown that the accuracy of the proposed method is higher than 90% [13].  
        In 2018, Song and their colleagues worked on classifying Facial Nerve Paralysis (FNP). They used CNN on the dataset of clinical images. Compared to the ground truth of the neurologists, their results matched the neurologists’ level and they have achieved 97% accuracy [14].
        Chandaliya et al. created a TeleStroke System (TSS). The sources of their dataset were Stroke Faces on Kaggle, YouTube-Facial-Palsy-Database, Yale Face Database. They used a simple Machine Learning algorithm i.e., Support Vector Machine (SVM) along with Deep Learning algorithms i.e., Fully Connected Neural Network, CNN, and VGG16 model to detect the stroke. They have found that the simple Machine Learning model gives better accuracy than the other Deep Learning models that are used in the experiment [15].
        In 2021, Gautam et al. worked on the classification of brain hemorrhagic and ischemic stroke using CNN. In this research, they used 13-layer CNN architecture on CT images acquired from the Himalayan Institute of Medical Sciences (HIMS), Dehradun, India.  Two experiments were conducted on two different datasets. In the first experiment, the CT image dataset is partitioned into 80% training and 20% testing sets. In the second experiment, 10-fold cross-validation has been performed on the dataset. The accuracy obtained from the first experiment on dataset 1 is 98.33% and in the second experiment is 98.77%. The accuracy obtained by applying CNN on dataset 2 in experiments 1 and 2 is 92.22% and 93.33% respectively. The result obtained from the proposed method was comparable to both AlexNet and ResNet50 [16].
       A research team built a Deep Learning IoT system for online stroke detection in skull computed tomography images. To identify whether it is a healthy brain, hemorrhagic stroke, or ischemic stroke, first, they extract the features through CNN and then classify using Machine Learning methods such as Bayesian Classifier, Multilayer Perceptron, k-Nearest Neighbor, Random Forest and Support Vector Machines. Furthermore, they made an Internet of Things (IoT) system to automate the diagnosis process. Their results have shown that CNN obtained 100% Accuracy, F1-Score, Recall
and Precision in combination with most of the classifiers tested which further proved the efficacy of their experiment [17].
       In 2020 Park et al. developed an automatic evaluation system, mFAST to examine the neurological deficits and personalized stroke prediction. They used mobile phones to monitor stroke patients with the help of sensors and machine learning classifiers (Naïve Bayes, SVM, Radial Basis Function (RBF)). The system could measure facial palsy using the phone's camera, arm weakness using inertial sensors, and speech disturbance using voice recorder and analyze the automatically collected data. Then it could make predictions based on those data. Performance metrics have shown the sensitivity, specificity, and Area Under the Curve (AUC) of three classes- ctrl, patients, and weighted average in three ML classifiers. For Naïve Bayes values of sensitivity of the three are 0.900, 0.938, 0.923, respectively. And specificity of the three is 0.937, 0.900, 0.914, respectively. The AUC of all three was 0.969. For SVM, values of sensitivity of the three are 0.9, 0.875, 0.885, respectively. And specificity of the three is 0.875, 0.900, 0.89, respectively.  The AUC of all three was 0.888.
 For RBF, values of sensitivity of the three are 0.9, 0.938, 0.923, respectively.  And specificity of the three is 0.937, 0.900, 0.914, respectively.  The AUC of all three was 0.944.
[18]
Ridha et al created a smart prediction system for detecting temporary facial paralysis which can perform physiotherapy routines under the supervision of physiotherapists remotely. They have created two bespoke Deep learning techniques for their program. The patient’s progress is measured utilizing the Google Machine Learning kit and Paralysis Prediction Neural Network (PPNN). Routine Time Suggestion Neural Network (RTSNN) is implemented so that the physiotherapists can accommodate more patients and update their physiotherapy routine treatment. They also built a phone application for user interface and headgear for physiotherapy.
[19]
       A group of researchers constructed a deep convolutional neural network (GoogleLeNet) to detect Unilateral Peripheral Facial Analysis (UPFP). This research has shown some promising results. The Deep Convolutional neural network produced classification accuracy of 91.25% for predicting the House-Brackmann degree on a given UPFP image dataset.
[20]
       In 2017, Yolcu et al. adopted deep learning approaches to monitor Neurological disorders through facial expressions. They utilized Convolutional Neural Network Architecture. Firstly the architecture segments the facial components known for facial expression recognition and forms an iconized image; then, it performs facial expression classification using obtained iconized facial components image combined with the raw facial images. The CNN has achieved 93.43% facial expression recognition accuracy, 6% more than the result obtained using the raw input images.
[21]
       A research team has built a cascaded encoder network structure to evaluate facial paralysis which explores the advantages of the facial semantic feature for facial spatial information extraction and then use that information to predict facial paralysis.  The first encoder is trained with the task of facial attribute semantic segmentation based on both normal face and paralysed face. The second encoder is trained with facial paralysis grading prediction task using paralysed facial images as training data. It can export facial paralysis features from the input facial images.  The accuracy, recall, confidence, and dice obtained from their model was 95.60%, 95.90%, 95.75%, 95.82% which was comparably better than the existing models [22,23] in the related areas.[24]

# CONCLUSION
In our work, we have focused on the detection of one of the most common symptoms of stroke, that is facial paralysis. For this, we have used a total of three deep learning models - VGG16, Inception V3, and ResNet50. According to the results, it is clearly evident that VGG16 stands out as the best model among the three. We plan on extending our research work in the future by using our models in the implementation of IOT devices and scalable mobile applications. Millions of people in rural areas in South Asian countries such as Bangladesh and India can highly benefit from these, as they do not have access to proper healthcare.

# REFERENCE:

[1]   "Stroke, heart disease top killers in Bangladesh", The Business Standard, 2021. [Online]. Available:
        https://www.tbsnews.net/bangladesh/health/stroke-heart-disease-top-killers-bangladesh-170065.
        [Accessed: 20- Jan- 2022].
[2]   G. Parra-Dominguez, R. Sanchez-Yanez and C. Garcia-Capulin, "Facial Paralysis Detection on Images 
        Using Key Point Analysis", Applied Sciences, vol. 11, no. 5, p. 2435, 2021. Available: 10.3390/app11052435.
[3]   Barbosa et al., "Efficient quantitative assessment of facial paralysis using iris segmentation and active 
        contour- based key points detection with hybrid classifier", BMC Medical Imaging, 
        vol.  16, no. 1, 2016.  Available: 10.1186/s12880-016-0117-0 [Accessed 21 February 2022].
[4]   C. Chang, M. Cheng and M. Ma, "Application of Machine Learning for Facial Stroke Detection", 
        2018 IEEE  23rd International Conference on Digital Signal Processing (DSP), 2018. 
        Available: 10.1109/icdsp.2018.8631568 [Accessed 21 February 2022].
[5]   C. Jiang et al., "Automatic Facial Paralysis Assessment via Computational Image Analysis", 
        Journal of Healthcare Engineering, vol. 2020, pp. 1-10, 2020. Available: 10.1155/2020/2398542.
[6]   T. Badriyah, N. Sakinah, I. Syarif and D. Syarif, "Machine Learning Algorithm for Stroke 
        Disease Classification", 2020 International Conference on Electrical, Communication, and 
        Computer Engineering (ICECCE), 2020. Available: 10.1109/icecce49384.2020.9179307 
        [Accessed 21 February 2022].
[7]   A. Gupta, “StrokeSave: A Novel, High-Performance Mobile Application for Stroke Diagnosis
        using Deep Learning and Computer Vision,” arXiv.org, 09-Jul-2019. [Online]. Available:
        https://arxiv.org/abs/1907.05358. [Accessed: 14-Feb-2022].
[8]   R. S. Jeena, A. Sukeshkumar, and K. Mahadevan, “Retina as a Biomarker of Stroke,”
        Computer-Aided Intervention and Diagnostics in Clinical and Medical Images, pp. 219–226, 2019.
[9]   D. Václavík, M. Bar, L. Klečka, D. Holeš, M. Čábal, and R. Mikulík, “Prehospital stroke scale
        (FAST PLUS Test) predicts patients with intracranial large vessel occlusion,” Brain and behavior,
        Sep-2018. [Online]. Available: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6160663/. [Accessed:
        14-Feb-2022].
[10] Y. Xue et al., "A multi-path 2.5 dimensional convolutional neural network system for segmenting stroke 
        Lesions in brain MRI images", NeuroImage: Clinical, vol. 25, p. 102118, 2020. 
        Available: 10.1016/j.nicl.2019.102118.
[11] The Warning Signs of Stroke. (2011, June 30). WebMD. https://www.webmd.com/stroke/guide/signs-of-stroke
[12] Gacek R. R. (1996). Hilger: "The Nature of Bell's Palsy." (Laryngoscope. 1949;59:228-235). 
        The Laryngoscope, 106(12 Pt 1), 1465–1468. https://doi.org/10.1097/00005537-199612000-00003
[13] Chin, C. L., Lin, B. J., Wu, G. R., Weng, T. C., Yang, C. S., Su, R. C., & Pan, Y. J. (2017, November). 
        An automated early ischemic stroke detection system using CNN deep learning algorithm. 
        In 2017 IEEE 8th   International Conference on Awareness Science and Technology 
        (iCAST) (pp. 368-372).   IEEE.
[14] Song, A., Wu, Z., Ding, X., Hu, Q., & Di, X. (2018). 
        Neurologist standard classification of facial nerve paralysis with deep neural networks. 
        Future Internet, 10(11), 111.
[15] Chandaliya, R., Joshi, P., & Afli, H. (2020). TeleStroke System (TSS). In CERC (pp. 325-334).
[16] Gautam, A., & Raman, B. (2021). 
        Towards effective classification of brain hemorrhagic and ischemic stroke using CNN. 
        Biomedical Signal   Processing and Control, 63, 102178.
[17] Dourado Jr, C. M., da Silva, S. P. P., da Nobrega, R. V. M., Barros, A. C. D. S., Reboucas Filho, P. P., 
        & de     Albuquerque, V. H. C. (2019). 
        Deep learning IoT system for online stroke detection in skull computed tomography images. 
        Computer Networks, 152, 25-39.
[18] Park, Eunjeong, Taehwa Han, and Hyo Suk Nam. "mFAST: automatic stoke evaluation system for 
        time-critical treatment with multimodal feature collection and machine learning classification. " Proceedings  
        of the 2020 12th International Conference on Computer and Automation Engineering. 2020.
[19] A. M. Ridha, W. Shehieb, P. Yacoub, K. Al-Balawneh and K. Arshad, "Smart Prediction System for 
        Facial Paralysis," 2020 7th International Conference on Electrical and Electronics Engineering (ICEEE), 
        2020, pp. 321-327, doi: 10.1109/ICEEE49618.2020.9102600.
[20] Z. Guo et al., "Deep assessment process: Objective assessment process for unilateral peripheral facial 
        Paralysis via deep convolutional neural network," 2017 IEEE 14th International Symposium on 
        Biomedical Imaging (ISBI 2017), 2017, pp. 135-138, doi: 10.1109/ISBI.2017.7950486.
[21] G. Yolcu et al., "Deep learning-based facial expression recognition for monitoring neurological 
        disorders,"   2017 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2017, 
        pp. 1652- 1657, doi: 10.1109/BIBM.2017.8217907.
[22] Wang, T., Zhang, S., Liu, L. A., Wu, G., & Dong, J. (2019). Automatic facial paralysis evaluation 
        augmented by a cascaded encoder network structure. IEEE Access, 7, 135621-135631.
[23] I. Song, N. Y. Yen, J. Vong, J. Diederich, and P. Yellowlees, ‘‘Profiling
        bell’s palsy based on House-Brackmann score,’’ J. Artif. Intell. Soft Comput. Res., vol. 3, no. 1, 
        pp. 41–50, Dec. 2014.
[24] H. S. Kim, S. Y. Kim, Y. H. Kim, and K. S. Park, ‘‘A smartphone-based automatic diagnosis system for 
        facial nerve palsy,’’ Sensors, vol. 15, no. 10, pp. 26756–26768, Oct. 2015.
[25] "Stroke Symptoms: Using the FAST Method and More", Healthline, 2021. [Online]. Available:
        https://www.healthline.com/health/stroke/symptoms. [Accessed: 20- Jan- 2022].
[26] K. Mehta, “Facial_Droop_and_Facial_Paralysis_image,” Kaggle, 23-Aug-2019. [Online].
        Available: https://www.kaggle.com/kaitavmehta/facial-droop-and-facial-paralysis-image. 
        [Accessed: 20-Jan-2022].
[27] "UTK Face Cropped", Kaggle.com, 2022. [Online]. 
     Available: https://www.kaggle.com/abhikjha/utk-face- cropped. [Accessed: 20- Jan- 2022].




