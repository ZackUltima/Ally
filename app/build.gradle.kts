plugins {
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kover) apply false
    alias(libs.plugins.ktlint) apply false
}

// On the laptop, build output goes outside the OneDrive folder so OneDrive does not sync thousands of
// build files (F3a plan, Global Constraints). CI leaves ALLY_GRADLE_OUT unset.
System.getenv("ALLY_GRADLE_OUT")?.let { out ->
    allprojects {
        val name = path.removePrefix(":").replace(':', '/').ifEmpty { "root" }
        layout.buildDirectory.set(file("$out/$name"))
    }
}
