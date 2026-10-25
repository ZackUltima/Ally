// Gradle root for the Ally phone app (technical plan T1). Android modules join in F4.
pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

rootProject.name = "ally"

include(":fall:flow")
