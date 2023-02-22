# Create Python 2.7 Environment in Anaconda
    
    conda create -n project python=2.7 -y (1 time)

# Conda Activate Crime Environment
    
    conda activate project


# Install Required Packages (1 time)
    
    conda install --channel https://conda.anaconda.org/menpo opencv3
	pip install Django djangorestframework django-extensions django-crispy-forms matplotlib  scikit-plot




### navigate to project floder 

cd C:\Users\aerah\Desktop\abnormal Newcode
 
### Command to Run Server
    
    python manage.py runserver  or 

       python manage.py runserver  0.0.0.0:8000


## Visit the server

    http://127.0.0.1:8000/

