import { gql } from "graphql-tag";

// ============================================
// QUERIES
// ============================================

export const GET_SPACE_EVENTS = gql`
  query SpaceEvents($spaceId: ID!) {
    spaceEvents(spaceId: $spaceId) {
      id
      title
      description
      location
      startsAt
      endsAt
      maxAttendees
      status
      attendeeCount
      createdBy
      tags
    }
  }
`;

// ============================================
// MUTATIONS
// ============================================

export const CREATE_EVENT = gql`
  mutation CreateEvent($input: CreateEventInput!) {
    createEvent(input: $input) {
      id
    }
  }
`;

export const RESPOND_TO_EVENT = gql`
  mutation RespondToEvent($eventId: ID!, $status: AttendeeStatus!) {
    respondToEvent(eventId: $eventId, status: $status) {
      id
      status
    }
  }
`;

export const MARK_EVENT_COMPLETED = gql`
  mutation MarkEventCompleted($eventId: ID!) {
    markEventCompleted(eventId: $eventId) {
      id
      status
    }
  }
`;
