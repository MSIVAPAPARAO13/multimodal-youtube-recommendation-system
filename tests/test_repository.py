from db.repository import VideoRepository


def test_repository_count_and_crud(in_memory_db):
    """Test VideoRepository count, create, and delete_all methods with in-memory DB."""
    repo = VideoRepository(in_memory_db)

    # Initial count should be 0
    assert repo.count() == 0

    # Add videos
    v1 = repo.create(
        video_id="vid_001",
        title="First Video",
        description="First Description",
        embedding=[0.1] * 512
    )
    assert v1.video_id == "vid_001"
    assert repo.count() == 1

    v2 = repo.create(
        video_id="vid_002",
        title="Second Video",
        description="Second Description",
        embedding=[0.2] * 512
    )
    assert v2.video_id == "vid_002"
    assert repo.count() == 2

    # Query by video_id
    fetched = repo.get_by_video_id("vid_001")
    assert fetched is not None
    assert fetched.title == "First Video"

    # Delete single video
    deleted = repo.delete("vid_001")
    assert deleted is True
    assert repo.count() == 1

    # Delete non-existent video
    deleted_missing = repo.delete("vid_non_existent")
    assert deleted_missing is False
    assert repo.count() == 1

    # Delete all
    cleared = repo.delete_all()
    assert cleared == 1
    assert repo.count() == 0
