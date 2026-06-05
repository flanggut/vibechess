import Foundation

/// Launch configuration for the Python GUI backend process.
struct BackendProcessCommand: Equatable, Sendable {
    var executable: String
    var arguments: [String]
    var workingDirectory: String?
    var environment: [String: String]?

    init(
        executable: String,
        arguments: [String] = [],
        workingDirectory: String? = nil,
        environment: [String: String]? = nil
    ) {
        self.executable = executable
        self.arguments = arguments
        self.workingDirectory = workingDirectory
        self.environment = environment
    }

    /// Developer default used by the local SwiftUI app while launched from `swift/`.
    static let developmentDefault = BackendProcessCommand(
        executable: "uv",
        arguments: ["run", "tinychess", "gui-server"],
        workingDirectory: ".."
    )
}

enum BackendClientError: Error, CustomStringConvertible {
    case launchFailed(command: BackendProcessCommand, underlying: String)
    case closed
    case writeFailed(String)
    case readFailed(String)
    case processTerminated(exitCode: Int32?, stderr: String)
    case invalidResponseUTF8
    case invalidResponseLine(String, underlying: String)
    case responseIDMismatch(expected: BackendMessageID, actual: BackendMessageID?)
    case backendRejected(BackendResponse)

    var description: String {
        switch self {
        case let .launchFailed(command, underlying):
            return "failed to launch backend command \(command.executable): \(underlying)"
        case .closed:
            return "backend client is closed"
        case let .writeFailed(message):
            return "failed to write backend request: \(message)"
        case let .readFailed(message):
            return "failed to read backend response: \(message)"
        case let .processTerminated(exitCode, stderr):
            let codeText = exitCode.map(String.init) ?? "unknown"
            return "backend process terminated with exit code \(codeText): \(stderr)"
        case .invalidResponseUTF8:
            return "backend response was not valid UTF-8"
        case let .invalidResponseLine(line, underlying):
            return "backend response was not valid protocol JSON (\(underlying)): \(line)"
        case let .responseIDMismatch(expected, actual):
            return "backend response id mismatch: expected \(expected), got \(String(describing: actual))"
        case let .backendRejected(response):
            let error = response.error
            return "backend rejected request: \(error?.code ?? "unknown") \(error?.message ?? "")"
        }
    }
}

/// Minimal async JSON-lines subprocess client for `tinychess gui-server`.
actor BackendClient {
    private let command: BackendProcessCommand
    private let process: Process
    private let input: FileHandle
    private let output: FileHandle
    private let stderrCapture: BackendStderrCapture
    private var closed = false

    init(command: BackendProcessCommand = .developmentDefault) throws {
        self.command = command

        let process = Process()
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        let stderrCapture = BackendStderrCapture(handle: stderrPipe.fileHandleForReading)

        if command.executable.contains("/") {
            process.executableURL = URL(fileURLWithPath: command.executable)
            process.arguments = command.arguments
        } else {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [command.executable] + command.arguments
        }
        if let workingDirectory = command.workingDirectory {
            process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory, isDirectory: true)
        }
        if let environment = command.environment {
            var mergedEnvironment = ProcessInfo.processInfo.environment
            for (key, value) in environment {
                mergedEnvironment[key] = value
            }
            process.environment = mergedEnvironment
        }
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            stderrCapture.stop()
            throw BackendClientError.launchFailed(
                command: command,
                underlying: String(describing: error)
            )
        }

        self.process = process
        self.input = stdinPipe.fileHandleForWriting
        self.output = stdoutPipe.fileHandleForReading
        self.stderrCapture = stderrCapture
    }

    deinit {
        stderrCapture.stop()
        try? input.close()
        try? output.close()
        if process.isRunning {
            process.terminate()
        }
    }

    /// Send one request and await exactly one line-delimited response.
    func send(_ request: BackendRequest) async throws -> BackendResponse {
        guard !closed else {
            throw BackendClientError.closed
        }
        guard process.isRunning else {
            throw BackendClientError.processTerminated(
                exitCode: process.terminationStatus,
                stderr: stderrCapture.text()
            )
        }

        try write(request)
        let line = try readResponseLine()
        let response = try decodeResponse(line)
        guard response.id == request.id else {
            throw BackendClientError.responseIDMismatch(expected: request.id, actual: response.id)
        }
        guard response.ok else {
            throw BackendClientError.backendRejected(response)
        }
        return response
    }

    /// Return stderr captured from the backend so UI code can include diagnostics in logs.
    func capturedStderr() -> String {
        stderrCapture.text()
    }

    /// Terminate the backend process and close all pipes. Safe to call more than once.
    func close() {
        guard !closed else {
            return
        }
        closed = true
        stderrCapture.stop()
        try? input.close()
        try? output.close()
        if process.isRunning {
            process.terminate()
        }
    }

    private func write(_ request: BackendRequest) throws {
        do {
            var data = try JSONEncoder().encode(request)
            data.append(0x0A)
            try input.write(contentsOf: data)
        } catch {
            throw BackendClientError.writeFailed(String(describing: error))
        }
    }

    private func readResponseLine() throws -> Data {
        var line = Data()
        while true {
            let chunk: Data
            do {
                chunk = try output.read(upToCount: 1) ?? Data()
            } catch {
                throw BackendClientError.readFailed(String(describing: error))
            }
            if chunk.isEmpty {
                let exitCode = process.isRunning ? nil : process.terminationStatus
                throw BackendClientError.processTerminated(
                    exitCode: exitCode,
                    stderr: stderrCapture.text()
                )
            }
            if chunk[chunk.startIndex] == 0x0A {
                if line.last == 0x0D {
                    line.removeLast()
                }
                return line
            }
            line.append(chunk)
        }
    }

    private func decodeResponse(_ line: Data) throws -> BackendResponse {
        guard let text = String(data: line, encoding: .utf8) else {
            throw BackendClientError.invalidResponseUTF8
        }
        do {
            return try JSONDecoder().decode(BackendResponse.self, from: line)
        } catch {
            throw BackendClientError.invalidResponseLine(
                text,
                underlying: String(describing: error)
            )
        }
    }
}

private final class BackendStderrCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()
    private let handle: FileHandle

    init(handle: FileHandle) {
        self.handle = handle
        handle.readabilityHandler = { [weak self] readableHandle in
            let chunk = readableHandle.availableData
            guard !chunk.isEmpty else {
                return
            }
            self?.append(chunk)
        }
    }

    func append(_ chunk: Data) {
        lock.lock()
        data.append(chunk)
        lock.unlock()
    }

    func text() -> String {
        lock.lock()
        let snapshot = data
        lock.unlock()
        return String(data: snapshot, encoding: .utf8) ?? ""
    }

    func stop() {
        handle.readabilityHandler = nil
        try? handle.close()
    }
}
