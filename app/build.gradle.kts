plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.shuttletrack"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.shuttletrack"
        minSdk = 27
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    // อย่าบีบอัดไฟล์โมเดล เพื่อให้ map เข้าหน่วยความจำได้
    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
}
