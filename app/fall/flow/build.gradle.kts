// The fall state machine in plain Kotlin (technical plan section 6, T1): no Android, no threads, no real clock.
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
