/**
 * GraphQL types for matching — data proxied from Identity Matcher.
 */

export const matchTypeDefs = `#graphql

  # ── Types from Identity Matcher ──────────────────────────────────

  type Match {
    user: MatchUser!
    similarity: Float!
    breakdown: MatchBreakdown!
  }

  type MatchUser {
    id: ID!
    name: String!
    givenName: String!
    familyName: String!
    image: String
    gender: String
    birthdate: String!
  }

  type MatchBreakdown {
    psychological: Float!
    values: Float!
    interests: Float!
    behavioral: Float!
  }

  type ProfileStatus {
    hasAssessment: Boolean!
    hasProfile: Boolean!
    assessmentName: String
    completedAt: String
  }

  type AssessmentSection {
    section: String!
    questions: [AssessmentQuestion!]!
  }

  type AssessmentQuestion {
    id: String!
    type: String!
    text: String!
    options: [String!]
    scaleLabels: [String!]
    template: String
    placeholder: String
  }

  type SubmitAssessmentResult {
    success: Boolean!
    profileComplete: Boolean!
  }

  # ── Queries ────────────────────────────────────────────────────

  extend type Query {
    """
    Find compatible matches for the authenticated user.
    Data is fetched from Identity Matcher.
    """
    findMatches(limit: Int, gender: [String!], minAge: Int, maxAge: Int): [Match!]!

    """
    Get the authenticated user's profile status (assessment + embeddings).
    Data is fetched from Identity Matcher.
    """
    profileStatus: ProfileStatus!

    """
    Get the assessment questionnaire definition.
    Data is fetched from Identity Matcher.
    """
    assessmentQuestions: [AssessmentSection!]!
  }

  # ── Mutations ──────────────────────────────────────────────────

  extend type Mutation {
    """
    Submit a completed assessment for the authenticated user.
    Data is sent to Identity Matcher.
    """
    submitAssessment(answers: JSON!): SubmitAssessmentResult!
  }
`;
