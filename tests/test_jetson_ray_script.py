import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class JetsonRayScriptTests(unittest.TestCase):
    def test_lan_head_ip_is_resolved_over_control_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            fake_bin = temp_dir / "bin"
            fake_bin.mkdir()
            ssh_log = temp_dir / "ssh.log"
            status_count = temp_dir / "status-count"
            active_hosts = temp_dir / "active-hosts"
            fake_ssh = fake_bin / "ssh"
            fake_ssh.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\n' "$*" >> {ssh_log}
                    case "$*" in
                      *"python3 - <<'PY'"*)
                        printf '%s\\n' "192.168.2.142"
                        exit 0
                        ;;
                      *"ray status"*)
                        count="$(cat {status_count} 2>/dev/null || printf 0)"
                        count=$((count + 1))
                        printf '%s\\n' "$count" > {status_count}
                        if [ "$count" -ge 2 ]; then
                          printf '%s\\n' "Resources" "host_agx" "frame_worker"
                        fi
                        exit 0
                        ;;
                      *)
                        exit 0
                        ;;
                    esac
                    """
                ),
                encoding="utf-8",
            )
            fake_ssh.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "JETSON_RAY_ACTIVE_HOSTS_FILE": str(active_hosts),
            }
            env.pop("JETSON_AGX_LAN_HOST", None)
            env.pop("JETSON_RAY_HEAD_IP", None)
            env.pop("JETSON_RAY_HEAD_SSH", None)

            result = subprocess.run(
                ["bash", "tools/start_jetson_frame_ray.sh"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("[jetson-ray] starting AGX Ray head", result.stdout)
            self.assertEqual(active_hosts.read_text(encoding="utf-8").strip(), "agx,agx")
            ssh_calls = ssh_log.read_text(encoding="utf-8")
            self.assertIn(" agx ", f" {ssh_calls} ")
            self.assertIn("192.168.2.142", ssh_calls)
            self.assertNotIn("192.168.31.201", ssh_calls)


if __name__ == "__main__":
    unittest.main()
