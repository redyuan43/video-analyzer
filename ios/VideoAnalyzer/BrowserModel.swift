import Foundation

/// Bridges SwiftUI state and the UIKit web view controller.
///
/// The controller owns the actual WKWebView; this only carries the state the
/// SwiftUI chrome needs to render, plus one-shot commands going the other way.
@MainActor
final class BrowserModel: ObservableObject {

    @Published var isLoading = false
    @Published var failureMessage: String?
    @Published var isShowingSettings = false

    /// Set to the configured address; changing it reloads the web view.
    @Published var serverURL: URL = AppConfig.serverURL

    /// A video URL handed over by the share extension, consumed once the web
    /// view has picked it up.
    @Published var pendingSharedURL: String?

    /// Incremented to ask the web view to reload.
    @Published private(set) var reloadCounter = 0

    func reload() {
        failureMessage = nil
        reloadCounter += 1
    }

    func apply(serverURL newValue: URL) {
        guard newValue != serverURL else { return }
        AppConfig.serverURL = newValue
        serverURL = newValue
        failureMessage = nil
    }
}
