import SwiftUI

struct ContentView: View {

    @EnvironmentObject private var browser: BrowserModel

    var body: some View {
        ZStack(alignment: .top) {
            AnalyzerWebView()
                .ignoresSafeArea()

            if browser.isLoading {
                ProgressView()
                    .progressViewStyle(.linear)
                    .tint(.accentColor)
            }

            // The web UI moved its own tabs to the bottom on phone widths, so
            // the top-right corner is free for the one thing the web layer
            // cannot provide: pointing the app at a different server.
            settingsButton

            if let message = browser.failureMessage {
                FailureOverlay(message: message)
            }
        }
        .sheet(isPresented: $browser.isShowingSettings) {
            ServerSettingsView()
                .environmentObject(browser)
        }
        // A wrong or unreachable address is the one failure the user can
        // actually fix, so offer the fix instead of just reporting it.
        .onChange(of: browser.failureMessage) { _, message in
            guard message != nil, !browser.isShowingSettings else { return }
            browser.isShowingSettings = true
        }
    }

    private var settingsButton: some View {
        HStack {
            Spacer()
            Button {
                browser.isShowingSettings = true
            } label: {
                Image(systemName: "server.rack")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(width: 32, height: 32)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .tint(.primary)
            .accessibilityLabel("服务器设置")
            .padding(.trailing, 14)
            .padding(.top, 6)
        }
    }
}

/// Shown when the web view could not reach the analyzer at all.
private struct FailureOverlay: View {

    @EnvironmentObject private var browser: BrowserModel
    let message: String

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "wifi.exclamationmark")
                .font(.system(size: 34))
                .foregroundStyle(.secondary)

            Text("连接不上分析服务")
                .font(.headline)

            Text(message)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Text(browser.serverURL.absoluteString)
                .font(.system(.footnote, design: .monospaced))
                .foregroundStyle(.secondary)

            Text("确认 iPhone 的 Tailscale 已连接，且分析服务正在运行。")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            HStack(spacing: 12) {
                Button("重试") { browser.reload() }
                    .buttonStyle(.borderedProminent)
                Button("服务器设置") { browser.isShowingSettings = true }
                    .buttonStyle(.bordered)
            }
            .padding(.top, 4)
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
    }
}
