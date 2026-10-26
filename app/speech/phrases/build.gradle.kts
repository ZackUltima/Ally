// The help and low-day phrase checker in plain Kotlin (technical plan sections C7 and E6, T20, T21): no Android, no threads.
plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kover)
    alias(libs.plugins.ktlint)
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    testImplementation(libs.junit)
}

ktlint {
    version.set(libs.versions.ktlint.get())
}

kover {
    reports {
        verify {
            rule {
                minBound(80)
            }
        }
    }
}
