Welcome to the ising model simultion. This read will be a guide for you to navigate through this code. 

Section 1: How to use this program. 
    Overall this program is very simple. You don't have to do much to run it. You just need to fullfill these 5 steps.

    Step 1: install vs code and latest version of python

    Step 2: install the following libraires. Run the following command in your terminal
    > pip install imageio numpy scipy ipywidgets matplotlib

    Step 3: install the python extension and jupyter extension in vs code.

    Step 4: Go to ising_model.ipynb and read through it to edit any initial variables in the program.

    Step 5: run the jupyter library.

    That's it.  I would recommend first implenting section 1 and going through the ising_model.ipynb file one before looking at the next sections. These next sections are going to be helpful, if you would like to unlock some extra functionality of this program.



Section 2: How does my algorithm work?
    This is for people, who would like to use and understand my algorithm for updating the isign model. Currently, it is commented out in ising_model.ipynb and you can just have to uncomment it to use it. There is nothing more to it. But if you would like to learn how it works, then here is the idea. 

    I created a list of tuples, where the first element of that tuple would have a row and the socond element would have a column.  Then, I run a for loop with time as a variable. Each time, I pick the first element from that list and use metropolis algorithm on it. If the algorithm doesn't change anything. I remove that element from that list. 

    If the algorithm does work, then, it changes the energies of the nearest 4 neighbours of that element. So, I pick those 4 neighbours. And if they are already in the list, then, that's great. Otherwise I append them to the end of the list. Then, I remove the initial element from the list as well.

    Like this, my algorithm goes through a cycle, rather than covering things randomly. It first goes through all the elements in the lattice, applying metropolis on all of them starting from the top left to the bottom right. Then, it stores the rows and columns of all the element, whose energy got changed in the first iteration, due to their neighbours getting flipped. Then, it only applies metropolis on those elements and so on until the time ends or the list is completely empty.


Section 3: How to export the simulation
    So, let's see. If you would like to export this simulation. There are 2 ways, you can do it:
    1. In an html file
    2. In an mp4 file

    To do it in the html file, I have written a line of code near the bottom of the page of ising_model.ipynb. It looks like this:
    # ig.display_ising_sequence_html(images, folder_file_name = "my_frame/animation.html")
    Just uncomment it and that's it. 

    To export it as an mp4 is another story. 
    First, try running it directly, without doing anything. If it works, then awesome. If it says that you don't have ffmpeg. 
    Then, do the following

    For that, you will need to install a library in your program called ffmpeg. 
    To rum import cv2 on your computer, use the following instruction. Other than the normal package installation, you would also need to install 'ffmpeg' 

    To do that, first install chocolatey

    1. Open Command Prompt as Administrator: Right-click on the Start button and select "Command Prompt (Admin)" 
    or "Windows PowerShell (Admin)".

    2. Run the Installation Command: Copy and paste the following command into the Command Prompt and press Enter:
    > @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"

    3. Wait for Installation: The installation process may take a few moments to complete. Once done, Chocolatey should be installed on your system.

    4. Verify Installation: After the installation is complete, you can verify that Chocolatey is installed correctly by running:
    > choco -?
    Now, you have successfully installed chocolatey

    5. Once, this is done, restart vs code again. This time in Administrator mode. 
    Open a New Terminal Window:

    6. Either press Ctrl + Shift + ``, or go to the menu and select Terminal>New Terminal`.
    This will open a new terminal window at the bottom of the Visual Studio Code interface.
    Select Command Prompt or PowerShell:

    7. By default, Visual Studio Code opens the terminal with PowerShell. Change it to use Command Prompt only for this step:
    Click on the dropdown arrow at the top-right corner of the terminal window.
    Select "Command Prompt" from the dropdown menu, depending on your preference.
    
    8.Verify Chocolatey Installation again:
    Once the terminal is open, you can verify that Chocolatey is installed by running:
    > choco -?
    If Chocolatey is installed correctly, you'll see the Chocolatey help message displayed in the terminal.

    
    then run the following command
    > choco install ffmpeg
    (if it shows, some error about file not being accessable then, make sure that vs code is open in Administrator mode)
    (if it shows some error about it already being there, but no instlled package, then force a re-install by using the following command:

    > choco install ffmpeg --force)

    Once it has successfully installed, close that terminal and open a python terminal and run the following Command:

    > pip install imageio[ffmpeg]   

    once you are done with all this, then, you can rum the export code to get a video out of it.





section 4:
    This section is for someone who would like to learn how to use the cython library, instead of the python library.
    Before you uncomment the cython library code, a couple things are needed. 

    Now, let's talk Cython. First of all, run the following command on your terminal
    
    > pip install cython. 

    Then, you will notice, 2 files in the ising_model folder
    1. ising_cython.pyx
    1. setup.py

    These 2 files create the cython library. Go to setup.py and run it. In order to run it for the first time and first time only, use this:
    > python setup.py build_ext --inplace

    Case - I (You were successfully able to run it)
    If you are able to run it, then awesome, running it will create 2 more files called:
    1. ising_cython.C
    2. ising_cython.cp312-win_amd64.pyd
    It will also create a folder called build

    Now, if you make any changes in the ising_cython.ipynb file, you will need to re-run the setup.py file to implement those changes, 
    but sometimes there is a problem in the '.pyd' file it creates. You see, when you re-run the code to update it. It is not able to overwrite the '.pyd' file and hence, it gives you a warning. It is able to overwrite the '.c' file but not '.pyd' file. So, your changes don't get implemented. To make sure that your changes get impletemented, you will have to delete the 'pyd' file manually. To do this, I would recommend that you use the following code:

    > rm -f ising_cython.cp312-win_amd64.pyd; clear; python setup.py build_ext --inplace
    
    This thing will forcefully delete the pyd file and then, rum your setup.py code properly.

    Once, you are good with all of this, then go to ising_model.ipynb and uncomment the ising_cython library and comment out the isinggame library and run the code.

    Case - II (You were not able to run it successfully and you get an error saying "Microsoft Visual C++ 14.0 or greater is required")
    You might encouter an error like this, so here is what you do. There are 2 solutions: 
    1. Run the following code in your terminal:
        Upgrade your pip with: python 
        > -m pip install --upgrade pip
        
        Upgrade your wheel with: pip install 
        > --upgrade wheel
        
        Upgrade your setuptools with: 
        > pip install --upgrade setuptools
        
        Close the terminal
        Try running the setup.py again
    
    2. install visual studio Build tools. Here is a stackoverflow link, that will provide you with a step by step guide on how to do it
        https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst

    Once, you were able to successfully run it, refer to case I.