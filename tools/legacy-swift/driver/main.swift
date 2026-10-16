// conformance-driver: run one contingency-py conformance fixture against the
// original Swift OperantKit and print the observed document.
//
// stdin:  a fixture JSON document ({"schedule": ..., "steps": [...]}).
// stdout: {"implementation": ..., "steps": [{"reinforced": ..., ...}, ...]}
//
// Each step is fed to the Swift schedule as
//   ResponseEntity(numOfResponses: <cumulative responses>, milliseconds: round(now * 1000))
// including steps whose "event" is null (time-based schedules decide on those).
// The Swift library has no reinforcer object, so on a reinforced step the driver
// reports {"time": now, "magnitude": 1.0, "label": "SR+"}.
//
// Variable schedules (VI / VR / VT) need "params.values_ms" (VI, VT) or
// "params.values" (VR): the Swift original draws its sequence from
// Int.random (no seed), so the caller injects the value list instead.
//
// This file is copied into a scratch clone of the Swift package as the
// executable target "conformance-driver". It is never compiled in place.

import Foundation
import OperantKit
import RxBlocking
import RxSwift

struct DriverError: Error, CustomStringConvertible {
    let description: String
}

func ms(_ seconds: Any?) throws -> Int {
    guard let s = seconds as? Double else {
        throw DriverError(description: "expected a number of seconds, got \(String(describing: seconds))")
    }
    return Int((s * 1000).rounded())
}

func ints(_ value: Any?, _ name: String) throws -> [Int] {
    guard let list = value as? [Any] else {
        throw DriverError(description: "missing integer list params.\(name)")
    }
    return try list.map { item in
        guard let n = item as? NSNumber else {
            throw DriverError(description: "params.\(name) holds a non-number")
        }
        return n.intValue
    }
}

func build(_ spec: [String: Any]) throws -> ScheduleUseCase {
    guard let type = spec["type"] as? String else {
        throw DriverError(description: "schedule without type")
    }
    let params = spec["params"] as? [String: Any] ?? [:]
    switch type {
    case "FR":
        guard let n = (params["n"] as? NSNumber)?.intValue else {
            throw DriverError(description: "FR needs params.n")
        }
        return FR(n)
    case "CRF":
        return CRF()
    case "EXT":
        return EXT()
    case "FI":
        return FI(repository: ScheduleRespositoryImpl(value: try ms(params["interval"])))
    case "FT":
        return FT(repository: ScheduleRespositoryImpl(value: try ms(params["interval"])))
    case "VI":
        return VI(try ms(params["mean_interval"]), values: try ints(params["values_ms"], "values_ms"))
    case "VT":
        return VT(try ms(params["mean_interval"]), values: try ints(params["values_ms"], "values_ms"))
    case "VR":
        let mean = (params["mean"] as? NSNumber)?.intValue ?? 0
        return VR(mean, values: try ints(params["values"], "values"))
    case "Alternative":
        guard let first = params["first"] as? [String: Any],
              let second = params["second"] as? [String: Any] else {
            throw DriverError(description: "Alternative needs params.first and params.second")
        }
        return Alt(try build(first), try build(second))
    default:
        throw DriverError(description: "schedule type \(type) is not available in the Swift original")
    }
}

func decide(_ schedule: ScheduleUseCase, _ entity: ResponseEntity, update: Bool) throws -> Bool {
    let result = try schedule.decision(entity, isUpdateIfReinforcement: update)
        .asObservable()
        .toBlocking()
        .single()
    return result.isReinforcement
}

func run() throws -> [String: Any] {
    let input = FileHandle.standardInput.readDataToEndOfFile()
    guard let fixture = try JSONSerialization.jsonObject(with: input) as? [String: Any],
          let spec = fixture["schedule"] as? [String: Any],
          let steps = fixture["steps"] as? [[String: Any]] else {
        throw DriverError(description: "stdin is not a fixture document")
    }
    let schedule = try build(spec)
    let subSchedules = (schedule as? AlternativeScheduleUseCase)?.subSchedules ?? []

    var responses = 0
    var observed: [[String: Any]] = []
    for step in steps {
        guard let now = step["now"] as? Double else {
            throw DriverError(description: "step without now")
        }
        if let event = step["event"], !(event is NSNull) {
            responses += 1
        }
        let entity = ResponseEntity(numOfResponses: responses, milliseconds: try ms(now))

        // Alternative: the Swift original only reports a Bool. Which component
        // fired is read from the components themselves (without updating them)
        // before the compound decision runs.
        var winner: String?
        for (index, sub) in subSchedules.enumerated() where winner == nil {
            if try decide(sub, entity, update: false) {
                winner = index == 0 ? "first" : "second"
            }
        }

        var out: [String: Any] = ["reinforced": false]
        if try decide(schedule, entity, update: true) {
            out["reinforced"] = true
            out["reinforcer"] = ["time": now, "magnitude": 1.0, "label": "SR+"]
            if let winner = winner {
                out["meta"] = ["alternative_winner": winner]
            }
        }
        observed.append(out)
    }
    return ["implementation": "OperantKit (Swift original) conformance-driver", "steps": observed]
}

do {
    let document = try run()
    let data = try JSONSerialization.data(withJSONObject: document)
    FileHandle.standardOutput.write(data)
} catch {
    FileHandle.standardError.write("conformance-driver: \(error)\n".data(using: .utf8)!)
    exit(2)
}
