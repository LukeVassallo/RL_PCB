pipeline {
    agent any

    stages {
        stage("Setup Environment") {
            steps {
                echo "Building the library"
                //sh 'make library'
            }
        }

        stage("Execute Experiment") {
            steps {
                echo "Building the application"
                //sh 'make test_app'
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

