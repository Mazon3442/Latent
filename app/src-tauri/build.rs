fn main() {
    // Distro packages (packaging/arch) set LATENT_PACKAGE, read with option_env! in main.rs.
    println!("cargo:rerun-if-env-changed=LATENT_PACKAGE");
    tauri_build::build()
}
