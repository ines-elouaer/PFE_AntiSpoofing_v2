from src.pad_system.challenge_manager import ChallengeManager


def main():
    manager = ChallengeManager(ttl_seconds=60)

    print("\n=== CREATE CHALLENGE ===")
    challenge = manager.create_challenge(source="test_challenge_manager")
    print(challenge)

    session_id = challenge["session_id"]
    challenge_id = challenge["challenge_id"]
    challenge_type = challenge["challenge_type"]

    print("\nSession ID    :", session_id)
    print("Challenge ID  :", challenge_id)
    print("Challenge Type:", challenge_type)

    print("\n=== VALIDATE CHALLENGE ===")
    result = manager.validate_challenge(challenge_id)
    print(result)

    assert result["is_valid"] is True
    assert result["reason"] == "challenge_valid"

    print("\n=== MARK USED ===")
    used = manager.mark_used(challenge_id)
    print(used)

    assert used["success"] is True
    assert used["reason"] == "challenge_marked_as_used"

    print("\n=== VALIDATE AGAIN ===")
    result2 = manager.validate_challenge(challenge_id)
    print(result2)

    assert result2["is_valid"] is False
    assert result2["reason"] == "challenge_already_used"

    print("\n[OK] ChallengeManager fonctionne correctement.")


if __name__ == "__main__":
    main()