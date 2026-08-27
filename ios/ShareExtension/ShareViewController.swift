import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// Share sheet entry point: turns a shared video link into an analyzer job.
///
/// The extension posts to the API itself rather than bouncing through the
/// app. Opening the containing app from a share extension relies on a
/// responder-chain trick that Apple has never sanctioned, and a one-tap
/// confirm sheet is better UX than an app switch anyway.
final class ShareViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()

        Task {
            let shared = await extractSharedURL()
            await MainActor.run { present(sharedURL: shared) }
        }
    }

    private func present(sharedURL: String?) {
        let view = ShareComposeView(
            sharedURL: sharedURL,
            onCancel: { [weak self] in self?.finish() },
            onDone: { [weak self] in self?.finish() }
        )

        let host = UIHostingController(rootView: view)
        addChild(host)
        host.view.frame = self.view.bounds
        host.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        host.view.backgroundColor = .clear
        self.view.addSubview(host.view)
        host.didMove(toParent: self)
    }

    private func finish() {
        extensionContext?.completeRequest(returningItems: nil)
    }

    // MARK: - Input extraction

    /// Pulls a usable http(s) URL out of whatever the sharing app provided.
    private func extractSharedURL() async -> String? {
        let items = (extensionContext?.inputItems as? [NSExtensionItem]) ?? []

        for item in items {
            for provider in item.attachments ?? [] {
                if let url = await load(UTType.url, from: provider) as? URL,
                   url.scheme == "http" || url.scheme == "https" {
                    return url.absoluteString
                }
                // Fall back to text, which is how several apps share links.
                if let text = await load(UTType.plainText, from: provider) as? String,
                   let found = Self.firstWebURL(in: text) {
                    return found
                }
            }
        }
        return nil
    }

    private func load(_ type: UTType, from provider: NSItemProvider) async -> Any? {
        guard provider.hasItemConformingToTypeIdentifier(type.identifier) else { return nil }
        return try? await provider.loadItem(forTypeIdentifier: type.identifier)
    }

    private static func firstWebURL(in text: String) -> String? {
        let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue)
        let range = NSRange(text.startIndex..., in: text)
        let match = detector?.firstMatch(in: text, range: range)
        guard let url = match?.url, url.scheme == "http" || url.scheme == "https" else {
            return nil
        }
        return url.absoluteString
    }
}

// MARK: - Compose UI

private struct ShareComposeView: View {

    let sharedURL: String?
    let onCancel: () -> Void
    let onDone: () -> Void

    @State private var autoStart = true
    @State private var status: Status = .ready

    private enum Status: Equatable {
        case ready
        case submitting
        case succeeded(String)
        case failed(String)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("视频链接") {
                    if let sharedURL {
                        Text(sharedURL)
                            .font(.system(.footnote, design: .monospaced))
                            .lineLimit(4)
                    } else {
                        Text("没有从分享内容里找到视频链接")
                            .foregroundStyle(.secondary)
                    }
                }

                Section {
                    Toggle("创建后立即开始分析", isOn: $autoStart)
                        .disabled(status == .submitting)
                } footer: {
                    Text("关掉的话只建任务，稍后在 App 里手动开始。")
                }

                Section {
                    switch status {
                    case .ready, .failed:
                        EmptyView()
                    case .submitting:
                        HStack {
                            ProgressView()
                            Text("正在提交…").foregroundStyle(.secondary)
                        }
                    case let .succeeded(message):
                        Label(message, systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    }

                    if case let .failed(message) = status {
                        Label(message, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                    }
                } footer: {
                    Text("提交到 \(AppConfig.serverURL.absoluteString)")
                        .font(.caption)
                }
            }
            .navigationTitle("分析这个视频")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("取消", action: onCancel)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("创建任务") {
                        Task { await submit() }
                    }
                    .disabled(sharedURL == nil || status == .submitting)
                }
            }
        }
    }

    private func submit() async {
        guard let sharedURL else { return }
        status = .submitting
        do {
            let job = try await VideoAnalyzerClient()
                .createJob(videoURL: sharedURL, autoStart: autoStart)
            status = .succeeded(
                job.count > 1
                    ? "已创建 \(job.count) 个选集任务，首个任务：\(job.id)"
                    : "任务已创建：\(job.id)"
            )
            // Leave the confirmation on screen briefly; dismissing instantly
            // makes a successful submit indistinguishable from a no-op.
            try? await Task.sleep(for: .seconds(1.2))
            onDone()
        } catch {
            status = .failed(error.localizedDescription)
        }
    }
}
