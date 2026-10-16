// fh_probe: call the Swift original's Fleshler-Hoffman generators and print
// the raw (unsorted) lists as JSON.
//
// Build together with the original helper only (Foundation, no RxSwift):
//   swiftc <Sources/Common/Helpers/FleshlerHoffman.swift> fh_probe.swift -o fh_probe
// The executable needs a file named main.swift for top-level code, so the
// test copies this file to main.swift in a scratch directory before compiling.
//
// Usage: fh_probe <interval|ratio|hantula> <value> <n> <runs>
// Output: [[Int], ...]  (one list per run)

import Foundation

let args = CommandLine.arguments
guard args.count == 5,
      let value = Int(args[2]), let n = Int(args[3]), let runs = Int(args[4]) else {
    FileHandle.standardError.write("usage: fh_probe <interval|ratio|hantula> <value> <n> <runs>\n".data(using: .utf8)!)
    exit(2)
}

let fh = FleshlerHoffman()
var lists: [[Int]] = []
for _ in 0..<runs {
    switch args[1] {
    case "interval":
        lists.append(fh.generatedInterval(value: value, iterations: n))
    case "ratio":
        lists.append(fh.generatedRatio(value: value, iterations: n))
    case "hantula":
        lists.append(fh.hantula1991(value: value, number: n))
    default:
        FileHandle.standardError.write("unknown generator \(args[1])\n".data(using: .utf8)!)
        exit(2)
    }
}
let data = try JSONSerialization.data(withJSONObject: lists)
FileHandle.standardOutput.write(data)
