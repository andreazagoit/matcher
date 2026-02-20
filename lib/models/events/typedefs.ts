/**
 * GraphQL Schema for Events
 */

export const eventTypeDefs = `#graphql
  type Event {
    id: ID!
    spaceId: ID!
    title: String!
    description: String
    location: String
    coordinates: EventCoordinates
    startsAt: DateTime!
    endsAt: DateTime
    maxAttendees: Int
    tags: [String!]!
    status: EventStatus!
    createdBy: ID!
    createdAt: DateTime!
    updatedAt: DateTime!
    attendees: [EventAttendee!]!
    attendeeCount: Int!
    """Status of the currently authenticated user for this event (null if not authenticated or not RSVP'd)"""
    myAttendeeStatus: AttendeeStatus
    """The space this event belongs to"""
    space: Space
  }

  type EventCoordinates {
    lat: Float!
    lon: Float!
  }

  type EventAttendee {
    id: ID!
    eventId: ID!
    userId: ID!
    user: User
    status: AttendeeStatus!
    registeredAt: DateTime!
    attendedAt: DateTime
  }

  enum EventStatus {
    draft
    published
    cancelled
    completed
  }

  enum AttendeeStatus {
    going
    interested
    attended
  }

  input CreateEventInput {
    spaceId: ID!
    title: String!
    description: String
    location: String
    lat: Float
    lon: Float
    startsAt: String!
    endsAt: String
    maxAttendees: Int
    tags: [String!]
  }

  input UpdateEventInput {
    title: String
    description: String
    location: String
    lat: Float
    lon: Float
    startsAt: String
    endsAt: String
    maxAttendees: Int
    tags: [String!]
    status: EventStatus
  }

  extend type Query {
    """
    Get a single event by ID.
    """
    event(id: ID!): Event

    """
    Get events for a specific space.
    """
    spaceEvents(spaceId: ID!): [Event!]!

    """
    Get the authenticated user's upcoming events.
    """
    myUpcomingEvents: [Event!]!

    """
    Get attendees for an event.
    """
    eventAttendees(eventId: ID!): [EventAttendee!]!

    """
    Search upcoming events by tags.
    matchAll=true requires ALL tags, false requires at least one.
    """
    eventsByTags(tags: [String!]!, matchAll: Boolean): [Event!]!

    """
    Get recommended events based on behavioral similarity and tag overlap.
    Falls back to upcoming events for new users.
    """
    recommendedEvents(limit: Int): [Event!]!
  }

  extend type Mutation {
    """
    Create a new event in a space.
    """
    createEvent(input: CreateEventInput!): Event!

    """
    Update an existing event.
    """
    updateEvent(id: ID!, input: UpdateEventInput!): Event!

    """
    Respond to an event (going, interested).
    """
    respondToEvent(eventId: ID!, status: AttendeeStatus!): EventAttendee!

    """
    Mark an event as completed. Attendees with status 'going' become 'attended'.
    """
    markEventCompleted(eventId: ID!): Event!
  }
`;
