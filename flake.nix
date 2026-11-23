{
  description = "NeuroKit2 Multi-version Python development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems =
        function:
        nixpkgs.lib.genAttrs supportedSystems (
          system:
          function {
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      devShells = forAllSystems (
        { pkgs }:
        let
          # 32-bit openblas is standard for many Python scientific wheels
          compatibleOpenBlas = pkgs.openblas.override { blas64 = false; };

          # List of libraries that non-nix binaries (from uv/pip) often need
          runtimeLibs = with pkgs; [
            stdenv.cc.cc.lib
            zlib
            xz
            openssl
            libffi
            compatibleOpenBlas
            liblapack
            gfortran.cc.lib
            suitesparse
            glib
          ];

          makePythonShell =
            pythonPkg:
            pkgs.mkShell {
              buildInputs = [
                pythonPkg
                pkgs.uv
                pkgs.pkg-config
              ]
              ++ runtimeLibs;

              # Note: LD_LIBRARY_PATH is primarily for Linux.
              # macOS uses DYLD_LIBRARY_PATH, but SIP often blocks it.
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

              shellHook = ''
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH"
                export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
                export TERM="xterm-256color"

                echo ""
                echo "❄️  NeuroKit2 Dev Shell (${pythonPkg.pname}) activated."
                echo "---------------------------------------------------------"

                if [ ! -d ".venv" ]; then
                    echo "TIP: Run the following to set up your environment:"
                    echo "   uv sync --all-groups --all-extras && uv run pre-commit install"
                else
                    echo "Virtualenv detected. Use 'uv run <cmd>' or 'source .venv/bin/activate'"
                fi
                echo "---------------------------------------------------------"
                echo ""
              '';
            };
        in
        {
          default = makePythonShell pkgs.python313;

          py311 = makePythonShell pkgs.python311;
          py312 = makePythonShell pkgs.python312;
          py313 = makePythonShell pkgs.python313;
          py314 = makePythonShell pkgs.python314;
        }
      );
    };
}
