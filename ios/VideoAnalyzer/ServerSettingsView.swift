import SwiftUI

/// Lets the user point the app at a different analyzer instance.
///
/// Tailscale IPs are stable but not permanent, and the same build should work
/// against a second machine without a recompile — so the address is editable
/// and testable in place rather than baked in.
struct ServerSettingsView: View {

    @EnvironmentObject private var browser: BrowserModel
    @Environment(\.dismiss) private var dismiss

    @State private var address: String = AppConfig.serverURL.absoluteString
    @State private var probe: ProbeState = .idle

    private enum ProbeState: Equatable {
        case idle
        case running
        case reachable
        case failed(String)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("http://100.x.y.z:5000", text: $address)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .font(.system(.body, design: .monospaced))
                        .onChange(of: address) { _, _ in probe = .idle }
                } header: {
                    Text("分析服务地址")
                } footer: {
                    Text("填 Tailscale 地址即可，例如 100.91.42.28:5000 或 ai-x10drg:5000。省略 http:// 会自动补上。")
                }

                Section {
                    Button {
                        Task { await testConnection() }
                    } label: {
                        HStack {
                            Text("测试连接")
                            Spacer()
                            probeIndicator
                        }
                    }
                    .disabled(probe == .running || AppConfig.normalizeServerURL(address) == nil)

                    if case let .failed(message) = probe {
                        Text(message)
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }
                }

                Section {
                    Button("恢复默认地址") {
                        address = AppConfig.defaultServerURL.absoluteString
                    }
                } footer: {
                    Text("默认地址：\(AppConfig.defaultServerURL.absoluteString)")
                }
            }
            .navigationTitle("服务器")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("取消") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("保存") { save() }
                        .disabled(AppConfig.normalizeServerURL(address) == nil)
                }
            }
        }
    }

    @ViewBuilder
    private var probeIndicator: some View {
        switch probe {
        case .idle:
            EmptyView()
        case .running:
            ProgressView()
        case .reachable:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed:
            Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
        }
    }

    private func testConnection() async {
        guard let url = AppConfig.normalizeServerURL(address) else { return }
        probe = .running
        do {
            let ok = try await VideoAnalyzerClient(baseURL: url).checkHealth()
            // A reachable-but-unhealthy server usually means the status service
            // is mid-restart, which is worth saying out loud.
            probe = ok ? .reachable : .failed("服务已响应，但报告自身状态异常")
        } catch {
            probe = .failed(error.localizedDescription)
        }
    }

    private func save() {
        guard let url = AppConfig.normalizeServerURL(address) else { return }
        browser.apply(serverURL: url)
        browser.reload()
        dismiss()
    }
}
