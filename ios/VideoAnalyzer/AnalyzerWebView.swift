import SwiftUI

/// SwiftUI wrapper around `AnalyzerWebViewController`.
struct AnalyzerWebView: UIViewControllerRepresentable {

    @EnvironmentObject private var browser: BrowserModel

    func makeUIViewController(context: Context) -> AnalyzerWebViewController {
        let controller = AnalyzerWebViewController(serverURL: browser.serverURL)

        // SwiftUI forbids mutating observed state during a view update, and
        // WebKit fires these callbacks synchronously from inside one.
        controller.onLoadingChanged = { isLoading in
            Task { @MainActor in browser.isLoading = isLoading }
        }
        controller.onFailure = { message in
            Task { @MainActor in browser.failureMessage = message }
        }

        context.coordinator.lastReloadCounter = browser.reloadCounter
        return controller
    }

    func updateUIViewController(_ controller: AnalyzerWebViewController, context: Context) {
        controller.serverURL = browser.serverURL

        if context.coordinator.lastReloadCounter != browser.reloadCounter {
            context.coordinator.lastReloadCounter = browser.reloadCounter
            controller.reload()
        }

        if let shared = browser.pendingSharedURL {
            controller.loadSharedVideoURL(shared)
            // Clear it so a later unrelated update does not reload the share.
            Task { @MainActor in browser.pendingSharedURL = nil }
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    final class Coordinator {
        var lastReloadCounter = 0
    }
}
