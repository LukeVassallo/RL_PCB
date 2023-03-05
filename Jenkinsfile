pipeline {
    agent any

    stages {
        stage("Setup Environment") {
            steps {
                PYTHON_VERSION=3.8.3
                GITLAB_SERVER=gitlab.lukevassallo.com
                echo "Building the library"

                #source ./setup.sh
                . ./setup.sh
                echo $RL_PCB

                # Get python 
                wget --no-verbose ftp://${GITLAB_SERVER}/Python-${PYTHON_VERSION}.tar.gz
                wget --no-verbose ftp://${GITLAB_SERVER}/Python-${PYTHON_VERSION}.tar.gz.md5

                md5sum -c Python-${PYTHON_VERSION}.tar.gz.md5

                tar -xf Python-${PYTHON_VERSION}.tar.gz

                #Create virtual environment
                python3 -m virtualenv venv --python="./Python-${PYTHON_VERSION}/python"
                source venv/bin/activate 
                which python
                python -c "import sys; print(sys.path)"
                python -V

                python -m pip install --upgrade pip

                # Install dependencies
                pip install numpy==1.23.3 matplotlib pyglet optuna tensorboard reportlab py-cpuinfo psutil pandas seaborn pynvml plotly moviepy
                pip install -U kaleido

                # Install my libraries
                python -m pip install ./lib/pcb_netlist_graph-0.0.1-py3-none-any.whl
                python -m pip install ./lib/pcb_file_io-0.0.1-py3-none-any.whl

                # Install pytorch 
                pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu

                # Check environment
                python test_setup.py

                # Deactivate environment
                deactivate
            }
        }

        stage("Execute Experiment") {
            steps {
                echo "Building the application"
                //sh 'make test_app'
                echo $RL_PCB
                source venv/bin/activate
                # Check environment
                python test_setup.py

                # Deactivate environment
                deactivate
            }
        }

        stage("Generate Training Report") {
            steps {
                echo "Testing the application"
                //sh 'make run_test_app'
            }
        }
        stage("Evaluation") {
            steps {
                echo "Testing the application"
                //sh 'make run_test_app'
            }
        }
        stage("Generate Evaluation Report") {
            steps {
                echo "Testing the application"
                //sh 'make run_test_app'
            }
        }
    }
    post {
        always {
            echo "Always"
        }
        success {
            echo "Success"
        }
        failure {
            echo "Failure"
        }
    }
}

