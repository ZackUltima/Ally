// The emotion level rules in plain Kotlin (technical plan section E6, T13): no Android, no threads, no real clock.
plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kover)
    alias(libs.plugins.ktlint)
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(project(":fall:flow"))
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
