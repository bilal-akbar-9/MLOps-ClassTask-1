pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-credentials')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Login to DockerHub') {
            steps {
                sh 'echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin'
            }
        }

        stage('Build and Push Docker image') {
            steps {
                script {
                    def dockerImage = docker.build("${env.DOCKERHUB_CREDENTIALS_USR}/ml-sentiment-app:latest")
                    dockerImage.push()
                }
            }
        }
    }

    post {
        always {
            sh 'docker logout'
        }
    }
}