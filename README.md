# Visionnairy.ai-ISP-project. 

a little piece of my graduate project. 

my project was translating [Visionary.AI](https://www.visionary.ai/) company image processing code from Python to C++ and CUDA.  

Since the main part of my project is protected by NDA, I can't share it. 

Here there are only additions that were used but the magic algorithm itself is under protection. 

1. Implementing convolution in Cuda 
    in my project I used OpenCV with the Cuda support module but, their convolution didn't give me the desired results.
    so I wrote by myself a kernel function to do that.  
2. Pool design pattern 
   in order to fast the process, I wanted to avoid memory allocation as it as possible. 
   so I did a GPU mats collection([a pool design pattern](https://www.geeksforgeeks.org/object-pool-design-pattern/)).
3. some usage with the pool, openCV with Cuda, and my convolution
