import Foundation
import Testing
@testable import VibeChessMacApp

@Test func backendProcessDevelopmentDefaultRunsFromRepositoryRoot() {
    #expect(BackendProcessCommand.developmentDefault.executable == "uv")
    #expect(BackendProcessCommand.developmentDefault.arguments == ["run", "vibechess", "gui-server"])
    #expect(BackendProcessCommand.developmentDefault.workingDirectory == "..")
}

@Test func backendClientSendsRequestAndDecodesResponse() async throws {
    let client = try BackendClient(
        command: shellCommand(
            """
            IFS= read -r request
            printf '%s\n' '{"id":"hello-1","ok":true,"version":"mock-backend"}'
            """
        )
    )
    defer {
        Task { await client.close() }
    }

    let response = try await client.send(
        BackendRequest(id: .string("hello-1"), cmd: .hello)
    )

    #expect(response.ok)
    #expect(response.version == "mock-backend")
}

@Test func backendClientThrowsForBackendErrorResponse() async throws {
    let client = try BackendClient(
        command: shellCommand(
            """
            IFS= read -r request
            printf '%s\n' '{"id":2,"ok":false,"error":{"code":"illegal_move","message":"nope"}}'
            """
        )
    )
    defer {
        Task { await client.close() }
    }

    var rejected: BackendResponse?
    do {
        _ = try await client.send(BackendRequest(id: .int(2), cmd: .makeMove, move: "e2e5"))
        Issue.record("Expected backendRejected error")
    } catch BackendClientError.backendRejected(let response) {
        rejected = response
    } catch {
        Issue.record("Unexpected error: \(error)")
    }

    #expect(rejected?.error?.code == "illegal_move")
    #expect(rejected?.error?.message == "nope")
}

@Test func backendClientThrowsForInvalidJSONLine() async throws {
    let client = try BackendClient(
        command: shellCommand(
            """
            IFS= read -r request
            printf '%s\n' 'not-json'
            """
        )
    )
    defer {
        Task { await client.close() }
    }

    var invalidLine: String?
    do {
        _ = try await client.send(BackendRequest(id: .int(3), cmd: .state))
        Issue.record("Expected invalidResponseLine error")
    } catch BackendClientError.invalidResponseLine(let line, _) {
        invalidLine = line
    } catch {
        Issue.record("Unexpected error: \(error)")
    }

    #expect(invalidLine == "not-json")
}

@Test func backendClientThrowsForResponseIDMismatch() async throws {
    let client = try BackendClient(
        command: shellCommand(
            """
            IFS= read -r request
            printf '%s\n' '{"id":"wrong","ok":true}'
            """
        )
    )
    defer {
        Task { await client.close() }
    }

    var actualID: BackendMessageID?
    do {
        _ = try await client.send(BackendRequest(id: .string("expected"), cmd: .state))
        Issue.record("Expected responseIDMismatch error")
    } catch BackendClientError.responseIDMismatch(_, let actual) {
        actualID = actual
    } catch {
        Issue.record("Unexpected error: \(error)")
    }

    #expect(actualID == .string("wrong"))
}

@Test func backendClientCapturesBackendStderr() async throws {
    let client = try BackendClient(
        command: shellCommand(
            """
            IFS= read -r request
            printf '%s\n' 'diagnostic from mock backend' >&2
            printf '%s\n' '{"id":4,"ok":true}'
            """
        )
    )
    defer {
        Task { await client.close() }
    }

    _ = try await client.send(BackendRequest(id: .int(4), cmd: .state))
    try await Task.sleep(nanoseconds: 50_000_000)

    let stderr = await client.capturedStderr()
    #expect(stderr.contains("diagnostic from mock backend"))
}

@Test func backendClientClosePreventsFurtherSends() async throws {
    let client = try BackendClient(
        command: shellCommand(
            """
            sleep 1
            """
        )
    )

    await client.close()

    var didThrowClosed = false
    do {
        _ = try await client.send(BackendRequest(id: .int(5), cmd: .state))
        Issue.record("Expected closed error")
    } catch BackendClientError.closed {
        didThrowClosed = true
    } catch {
        Issue.record("Unexpected error: \(error)")
    }

    #expect(didThrowClosed)
}

private func shellCommand(_ script: String) -> BackendProcessCommand {
    BackendProcessCommand(executable: "/bin/sh", arguments: ["-c", script])
}
